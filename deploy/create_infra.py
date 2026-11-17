#!/usr/bin/env python3
"""
Idempotent-ish bootstrap of the shared Runpod infrastructure via the v2
REST API (api.runpod.io/v2 — v1's rest.runpod.io was retired 2026-11-15):

    1. a container registry credential (for a private Docker Hub image)
    2. a Network Volume (persistent GGUF storage, survives worker restarts)
    3. a Serverless Endpoint (Flex workers, workers.min=0, attached to the volume)

v2 has no separate Template resource — image/env/ports go straight on the
endpoint, so this bootstrap is one resource shorter than it used to be.

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

API = "https://api.runpod.io/v2"
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


def gpu_pools_for(gpu_type_ids: list[str], api_key: str) -> list[str]:
    """Map literal GPU ids (e.g. "NVIDIA A100 80GB PCIe") to their v2
    catalog pool (e.g. "AMPERE_80") — v2 endpoints select on pools, not
    individual ids; see GET /v2/catalog/gpus."""
    catalog = api("GET", "/catalog/gpus", api_key, params={"product": "SERVERLESS"})["gpus"]
    by_id = {g["id"]: g.get("pool") for g in catalog}
    pools, seen = [], set()
    for gpu_id in gpu_type_ids:
        pool = by_id.get(gpu_id)
        if not pool:
            sys.exit(f"unknown or non-poolable Runpod GPU id: {gpu_id!r}")
        if pool not in seen:
            seen.add(pool)
            pools.append(pool)
    return pools


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="e.g. docker.io/you/gptoss-120b-worker:latest")
    p.add_argument("--name", default="gptoss-120b")
    p.add_argument("--data-center", default="US-CA-2")  # must support network volumes; see Runpod's /datacenters
    p.add_argument("--volume-size-gb", type=int, default=100)
    p.add_argument("--gpu-type-ids", nargs="+",
                    default=["NVIDIA A100 80GB PCIe", "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"])
    p.add_argument("--idle-timeout", type=int, default=60)
    p.add_argument("--workers-max", type=int, default=1)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--registry-username", default=os.environ.get("DOCKERHUB_USERNAME", ""))
    p.add_argument("--registry-password", default=os.environ.get("DOCKERHUB_TOKEN", ""))
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("set RUNPOD_API_KEY in your environment first")

    state = load_state()

    # 0. Container registry auth (required for private Docker Hub repos)
    if "registry_auth_id" in state:
        print(f"reusing existing container registry auth {state['registry_auth_id']}")
    elif args.registry_username and args.registry_password:
        reg = api("POST", "/registries", api_key, json={
            "name": f"{args.name}-dockerhub",
            "username": args.registry_username,
            "password": args.registry_password,
        })
        state["registry_auth_id"] = reg["id"]
        save_state(state)
        print(f"created container registry auth {reg['id']}")
    else:
        print("no registry credentials given; assuming the image is public")

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

    # 2. Endpoint (image/env/ports go straight on it — v2 has no separate template)
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

    if "endpoint_id" in state:
        print(f"reusing existing endpoint {state['endpoint_id']}")
    else:
        pools = gpu_pools_for(args.gpu_type_ids, api_key)
        ep_body = {
            "name": args.name,
            "image": args.image,
            "type": "QUEUE",
            "disk": 20,  # image itself is small; the model lives on the network volume
            "ports": ["8080/http"],
            "env": env,
            "gpu": {"pools": pools},
            "workers": {"min": 0, "max": args.workers_max, "idleTimeout": args.idle_timeout},  # min=0 -> scale-to-zero
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 1},
            "flashboot": "FLASHBOOT",
            "mounts": {"network": [{"volumeId": state["network_volume_id"]}]},
        }
        if "registry_auth_id" in state:
            ep_body["registry"] = state["registry_auth_id"]
        ep = api("POST", "/serverless", api_key, json=ep_body)
        state["endpoint_id"] = ep["id"]
        save_state(state)
        print(f"created endpoint {ep['id']}")

    print("\nDone. Set these in proxy/.env (or your deployment's env):")
    print(f"  RUNPOD_ENDPOINT_ID={state['endpoint_id']}")
    print(f"\nState saved to {STATE_FILE} (delete it to force re-creation of resources).")


if __name__ == "__main__":
    main()
