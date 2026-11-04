#!/usr/bin/env python3
"""
Idempotent-ish bootstrap of the Runpod infrastructure via the REST API:

    1. a Network Volume (persistent GGUF storage, survives worker restarts)
    2. a Serverless Template (docker image + env vars)
    3. a Serverless Endpoint (Flex workers, workersMin=0, attached to the volume)

Usage:
    export RUNPOD_API_KEY=...
    python3 deploy/create_infra.py \
        --image docker.io/<user>/gptoss-120b-worker:latest \
        --name gptoss-120b \
        --data-center US-KS-2 \
        --volume-size-gb 100 \
        --gpu-type-ids "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB PCIe"

Prints the resulting RUNPOD_ENDPOINT_ID (put it in your proxy's .env) and
writes deploy/infra_state.json so re-runs can reuse existing resources
instead of creating duplicates.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import httpx

API = "https://rest.runpod.io/v1"
STATE_FILE = Path(__file__).parent / "infra_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def api(method: str, path: str, api_key: str, **kwargs) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = httpx.request(method, f"{API}{path}", headers=headers, timeout=60, **kwargs)
    if r.status_code >= 400:
        print(f"ERROR {method} {path} -> {r.status_code}: {r.text}", file=sys.stderr)
        r.raise_for_status()
    return r.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="e.g. docker.io/you/gptoss-120b-worker:latest")
    p.add_argument("--name", default="gptoss-120b")
    p.add_argument("--data-center", default="US-KS-2")
    p.add_argument("--volume-size-gb", type=int, default=100)
    p.add_argument("--gpu-type-ids", nargs="+", default=["NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB PCIe"])
    p.add_argument("--idle-timeout", type=int, default=60)
    p.add_argument("--workers-max", type=int, default=1)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("set RUNPOD_API_KEY in your environment first")

    state = load_state()

    # 1. Network volume
    if "network_volume_id" in state:
        print(f"reusing existing network volume {state['network_volume_id']}")
    else:
        vol = api("POST", "/network-volumes", api_key, json={
            "name": f"{args.name}-model-storage",
            "dataCenter": args.data_center,
            "size": args.volume_size_gb,
        })
        state["network_volume_id"] = vol["id"]
        save_state(state)
        print(f"created network volume {vol['id']} ({args.volume_size_gb} GB in {args.data_center})")

    # 2. Template
    env = {
        "SERVED_MODEL_NAME": os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-uncensored"),
        "MODEL_REPO": os.environ.get("MODEL_REPO", "HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive"),
        "MODEL_FILE": os.environ.get("MODEL_FILE", "GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf"),
        "MODEL_DIR": "/runpod-volume/models/gpt-oss-120b-uncensored",
        "MAX_MODEL_LEN": os.environ.get("MAX_MODEL_LEN", "8192"),
        "N_GPU_LAYERS": os.environ.get("N_GPU_LAYERS", "999"),
        "N_CPU_MOE": os.environ.get("N_CPU_MOE", "0"),
        "MAX_CONCURRENT": os.environ.get("MAX_CONCURRENT", "2"),
    }
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token  # only ever sent over HTTPS into Runpod's own secret store

    if "template_id" in state:
        print(f"reusing existing template {state['template_id']}")
    else:
        tmpl = api("POST", "/templates", api_key, json={
            "name": f"{args.name}-template",
            "imageName": args.image,
            "category": "NVIDIA",
            "isServerless": True,
            "containerDiskInGb": 20,  # image itself is small; the model lives on the network volume
            "ports": ["8080/http"],
            "env": env,
        })
        state["template_id"] = tmpl["id"]
        save_state(state)
        print(f"created template {tmpl['id']}")

    # 3. Endpoint
    if "endpoint_id" in state:
        print(f"reusing existing endpoint {state['endpoint_id']}")
    else:
        ep = api("POST", "/endpoints", api_key, json={
            "name": args.name,
            "templateId": state["template_id"],
            "computeType": "GPU",
            "gpuTypeIds": args.gpu_type_ids,
            "gpuCount": 1,
            "workersMin": 0,          # <-- Active Workers = 0 (scale-to-zero)
            "workersMax": args.workers_max,
            "idleTimeout": args.idle_timeout,
            "flashboot": True,
            "networkVolumeId": state["network_volume_id"],
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 1,
        })
        state["endpoint_id"] = ep["id"]
        save_state(state)
        print(f"created endpoint {ep['id']}")

    print("\nDone. Set these in proxy/.env (or your deployment's env):")
    print(f"  RUNPOD_ENDPOINT_ID={state['endpoint_id']}")
    print(f"\nState saved to {STATE_FILE} (delete it to force re-creation of resources).")


if __name__ == "__main__":
    main()
