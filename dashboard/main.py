"""
Local control panel: shows Runpod balance/spend and Modal app status for
this deployment, with Start/Stop buttons for each provider.

Binds to 127.0.0.1 only by default (see scripts/run_dashboard.sh) — it
holds RUNPOD_API_KEY server-side and can spend money (recreating an
endpoint starts GPU billing again), so it must never be exposed beyond
localhost without adding real authentication first.

Run:
    set -a; . ./.env; set +a
    bash scripts/run_dashboard.sh
    # open http://localhost:8787
"""
import json
import logging
import os
import re
import subprocess
import urllib.parse
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dashboard")

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "deploy" / "infra_state.json"

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DOCKERHUB_USERNAME = os.environ.get("DOCKERHUB_USERNAME", "")
RUNPOD_DATA_CENTER = os.environ.get("RUNPOD_DATA_CENTER", "US-CA-2")
RUNPOD_NETWORK_VOLUME_SIZE_GB = os.environ.get("RUNPOD_NETWORK_VOLUME_SIZE_GB", "100")
RUNPOD_GPU_TYPE_IDS = [g for g in os.environ.get(
    "RUNPOD_GPU_TYPE_IDS",
    "NVIDIA A100 80GB PCIe,NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3",
).split(",") if g]

MODAL_APP_NAME = "gptoss-120b"
MODAL_MODEL_DIR = "/cache/models/gpt-oss-120b-uncensored"  # must match deploy/modal_deploy.sh
ENV_FILE = ROOT / ".env"

app = FastAPI(title="GPT-OSS-120B control panel")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def update_env_file(updates: dict[str, str]) -> None:
    """Rewrite matching KEY=... lines in .env in place; append any that
    weren't already there. Keeps every other line untouched."""
    if not ENV_FILE.exists():
        return
    lines = ENV_FILE.read_text().splitlines()
    seen = set()
    new_lines = []
    for line in lines:
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=", line)
        if m and m.group(1) in updates:
            key = m.group(1)
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


# ----------------------------------------------------------------- Model --

HF_FILE_RE = re.compile(r"huggingface\.co/([^/\s]+/[^/\s]+?)/(?:blob|resolve)/[^/]+/([^\s?]+)")
HF_REPO_RE = re.compile(r"^(?:https?://(?:www\.)?huggingface\.co/)?([\w.-]+/[\w.-]+?)(?:/tree/.*)?/?$")


def parse_hf_input(raw: str) -> tuple[str, str | None]:
    raw = raw.strip()
    m = HF_FILE_RE.search(raw)
    if m:
        return m.group(1), urllib.parse.unquote(m.group(2))
    m = HF_REPO_RE.match(raw)
    if m:
        return m.group(1), None
    raise ValueError(f"couldn't parse a Hugging Face repo/file from: {raw!r}")


async def find_gguf_file(repo_id: str) -> str:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"https://huggingface.co/api/models/{repo_id}")
        if r.status_code == 404:
            raise ValueError(f"repo not found on Hugging Face: {repo_id}")
        r.raise_for_status()
        siblings = r.json().get("siblings", [])
    ggufs = [s["rfilename"] for s in siblings if s["rfilename"].lower().endswith(".gguf")]
    if len(ggufs) == 1:
        return ggufs[0]
    if not ggufs:
        raise ValueError(f"no .gguf files found in {repo_id} (only GGUF models are supported here)")
    raise ValueError(
        f"{len(ggufs)} .gguf files found in {repo_id} — paste a direct link to the one you want, e.g. "
        f"https://huggingface.co/{repo_id}/resolve/main/{ggufs[0]}"
    )


def derive_served_name(repo_id: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", repo_id.split("/")[-1].lower()).strip("-")
    return name or "custom-model"


class ModelUpdateRequest(BaseModel):
    hf_url: str
    served_name: str | None = None


@app.get("/api/model/current")
async def model_current():
    return {
        "repo": os.environ.get("MODEL_REPO", ""),
        "file": os.environ.get("MODEL_FILE", ""),
        "served_name": os.environ.get("SERVED_MODEL_NAME", ""),
    }


@app.post("/api/model/update")
async def model_update(req: ModelUpdateRequest):
    try:
        repo_id, filename = parse_hf_input(req.hf_url)
        if filename is None:
            filename = await find_gguf_file(repo_id)
        elif not filename.lower().endswith(".gguf"):
            raise ValueError(f"only .gguf files are supported by llama-server here, got: {filename!r}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resolve_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.head(resolve_url)
        if r.status_code >= 400:
            raise HTTPException(
                status_code=400,
                detail=f"file not found on Hugging Face ({r.status_code}): {resolve_url}",
            )

    served_name = req.served_name.strip() if req.served_name and req.served_name.strip() else derive_served_name(repo_id)

    os.environ["MODEL_REPO"] = repo_id
    os.environ["MODEL_FILE"] = filename
    os.environ["SERVED_MODEL_NAME"] = served_name
    update_env_file({"MODEL_REPO": repo_id, "MODEL_FILE": filename, "SERVED_MODEL_NAME": served_name})

    result = {"repo": repo_id, "file": filename, "served_name": served_name,
              "note": "Existing warm workers won't pick this up until their next cold start "
                      "(Stop then Start to force it immediately)."}

    state = load_state()
    template_id = state.get("template_id")
    if template_id and RUNPOD_API_KEY:
        async with httpx.AsyncClient(timeout=20) as client:
            cur = await client.get(
                f"https://rest.runpod.io/v1/templates/{template_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
            if cur.status_code >= 400:
                result["runpod_template_error"] = cur.text
            else:
                env = cur.json().get("env", {})
                env.update({"MODEL_REPO": repo_id, "MODEL_FILE": filename, "SERVED_MODEL_NAME": served_name})
                patch = await client.patch(
                    f"https://rest.runpod.io/v1/templates/{template_id}",
                    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                    json={"env": env},
                )
                if patch.status_code >= 400:
                    result["runpod_template_error"] = patch.text
                else:
                    result["runpod_template_updated"] = True

    if os.environ.get("MODAL_TOKEN_ID"):
        cmd = [
            "python3", "-m", "modal", "secret", "create", "gptoss-worker-secrets",
            f"SERVED_MODEL_NAME={served_name}",
            f"MODEL_REPO={repo_id}",
            f"MODEL_FILE={filename}",
            f"MODEL_DIR={MODAL_MODEL_DIR}",
            f"MAX_MODEL_LEN={os.environ.get('MAX_MODEL_LEN', '8192')}",
            f"N_GPU_LAYERS={os.environ.get('N_GPU_LAYERS', '999')}",
            f"N_CPU_MOE={os.environ.get('N_CPU_MOE', '0')}",
            f"MAX_CONCURRENT={os.environ.get('MAX_CONCURRENT', '2')}",
            f"LLAMA_SERVER_PORT={os.environ.get('LLAMA_SERVER_PORT', '8080')}",
            f"LLAMA_INTERNAL_API_KEY={os.environ.get('LLAMA_INTERNAL_API_KEY', '')}",
            f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
            f"PROXY_API_KEY={os.environ.get('PROXY_API_KEY', '')}",
            f"ALLOWED_ORIGINS={os.environ.get('ALLOWED_ORIGINS', '')}",
            f"RATE_LIMIT_PER_MINUTE={os.environ.get('RATE_LIMIT_PER_MINUTE', '60')}",
            f"UPSTREAM_TIMEOUT_SECONDS={os.environ.get('UPSTREAM_TIMEOUT_SECONDS', '600')}",
            f"LOG_PROMPTS={os.environ.get('LOG_PROMPTS', 'false')}",
            "--force",
        ]
        mresult = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ), timeout=60)
        result["modal_secret_updated"] = mresult.returncode == 0
        if mresult.returncode != 0:
            result["modal_secret_error"] = (mresult.stderr or mresult.stdout)[-1000:]

    log.warning("Model switched to %s/%s (served as %s)", repo_id, filename, served_name)
    return result


# ---------------------------------------------------------------- Runpod --

async def runpod_account() -> dict:
    if not RUNPOD_API_KEY:
        return {"error": "RUNPOD_API_KEY not set"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"query": "query { myself { clientBalance currentSpendPerHr } }"},
        )
        r.raise_for_status()
        return r.json()["data"]["myself"]


async def runpod_workers(endpoint_id: str) -> dict | None:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"https://api.runpod.io/v2/serverless/{endpoint_id}/workers",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


@app.get("/api/runpod/status")
async def runpod_status():
    account = await runpod_account()
    state = load_state()
    endpoint_id = state.get("endpoint_id")
    workers = await runpod_workers(endpoint_id) if endpoint_id else None
    return {
        "configured": bool(RUNPOD_API_KEY),
        "account": account,
        "endpoint_id": endpoint_id,
        "deployed": endpoint_id is not None and workers is not None,
        "workers": workers.get("summary") if workers else None,
    }


@app.post("/api/runpod/stop")
async def runpod_stop():
    state = load_state()
    endpoint_id = state.get("endpoint_id")
    if not endpoint_id:
        return {"status": "already stopped"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.delete(
            f"https://rest.runpod.io/v1/endpoints/{endpoint_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        )
        if r.status_code not in (200, 204, 404):
            raise HTTPException(status_code=502, detail=f"Runpod delete failed: {r.text}")
    state.pop("endpoint_id", None)
    save_state(state)
    log.info("Runpod endpoint %s deleted", endpoint_id)
    return {"status": "stopped", "endpoint_id": endpoint_id}


@app.post("/api/runpod/teardown")
async def runpod_teardown():
    """Delete every paid Runpod resource for this deployment: endpoint,
    template, and the network volume (which holds the 61GB model — the
    next deploy will have to re-download it). Unlike /stop, this leaves
    nothing billing at all, not even the volume's storage fee."""
    state = load_state()
    deleted = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for key, path in (
            ("endpoint_id", "endpoints"),
            ("template_id", "templates"),
            ("network_volume_id", "networkvolumes"),
        ):
            resource_id = state.get(key)
            if not resource_id:
                continue
            r = await client.delete(
                f"https://rest.runpod.io/v1/{path}/{resource_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
            if r.status_code not in (200, 204, 404):
                raise HTTPException(status_code=502, detail=f"delete {path}/{resource_id} failed: {r.text}")
            deleted[key] = resource_id
    save_state({})  # registry_auth isn't billed; leave teardown of that to re-running deploy.sh if ever needed
    log.warning("Runpod full teardown: deleted %s", deleted)
    return {"status": "deleted", "deleted": deleted}


@app.post("/api/runpod/start")
async def runpod_start():
    state = load_state()
    if state.get("endpoint_id"):
        return {"status": "already running", "endpoint_id": state["endpoint_id"]}
    if not DOCKERHUB_USERNAME:
        raise HTTPException(status_code=400, detail="DOCKERHUB_USERNAME not set")
    cmd = [
        "python3", "-I", str(ROOT / "deploy" / "create_infra.py"),
        "--image", f"docker.io/{DOCKERHUB_USERNAME}/gptoss-120b-worker:latest",
        "--name", "gptoss-120b",
        "--data-center", RUNPOD_DATA_CENTER,
        "--volume-size-gb", RUNPOD_NETWORK_VOLUME_SIZE_GB,
        "--gpu-type-ids", *RUNPOD_GPU_TYPE_IDS,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=os.environ, timeout=120)
    if result.returncode != 0:
        raise HTTPException(status_code=502, detail=result.stdout[-2000:] + result.stderr[-2000:])
    new_state = load_state()
    log.info("Runpod endpoint recreated: %s", new_state.get("endpoint_id"))
    return {"status": "started", "endpoint_id": new_state.get("endpoint_id"), "log": result.stdout[-2000:]}


# ------------------------------------------------------------------ Modal --

def modal_env() -> dict:
    env = dict(os.environ)
    return env


@app.get("/api/modal/status")
async def modal_status():
    if not os.environ.get("MODAL_TOKEN_ID"):
        return {"configured": False}
    try:
        result = subprocess.run(
            ["python3", "-m", "modal", "app", "list", "--json"],
            cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        return {"configured": True, "error": str(e)}
    if result.returncode != 0:
        return {"configured": True, "error": (result.stderr or result.stdout)[-1000:]}
    try:
        apps = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"configured": True, "error": "could not parse `modal app list` output"}
    mine = [a for a in apps if a.get("Name") == MODAL_APP_NAME or a.get("name") == MODAL_APP_NAME]
    return {"configured": True, "apps": mine, "deployed": any(
        (a.get("State") or a.get("state") or "").lower() in ("deployed", "running") for a in mine
    )}


@app.post("/api/modal/stop")
async def modal_stop():
    result = subprocess.run(
        ["python3", "-m", "modal", "app", "stop", MODAL_APP_NAME, "-y"],
        cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=502, detail=(result.stderr or result.stdout)[-2000:])
    log.info("Modal app %s stopped", MODAL_APP_NAME)
    return {"status": "stopped", "log": result.stdout[-1000:]}


@app.post("/api/modal/teardown")
async def modal_teardown():
    """Stop the app and delete every paid/stored Modal resource: the model
    volume (61GB — next deploy re-downloads it) and the two secrets."""
    steps = [
        ["python3", "-m", "modal", "app", "stop", MODAL_APP_NAME, "-y"],
        ["python3", "-m", "modal", "volume", "delete", os.environ.get("MODAL_VOLUME_NAME", "gptoss-120b-model-cache"), "-y"],
        ["python3", "-m", "modal", "secret", "delete", "gptoss-worker-secrets", "-y"],
        ["python3", "-m", "modal", "secret", "delete", "dockerhub-registry-secret", "-y"],
    ]
    logs = []
    for cmd in steps:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60)
        logs.append(f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}")
        # Don't hard-fail if a given resource was already gone / never existed.
    state_file = ROOT / "deploy" / "modal_state.json"
    if state_file.exists():
        state_file.unlink()
    log.warning("Modal full teardown complete")
    return {"status": "deleted", "log": "\n---\n".join(logs)[-4000:]}


@app.post("/api/modal/start")
async def modal_start():
    result = subprocess.run(
        ["bash", str(ROOT / "deploy" / "modal_deploy.sh")],
        cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=600,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=502, detail=(result.stderr or result.stdout)[-3000:])
    log.info("Modal app %s (re)deployed", MODAL_APP_NAME)
    return {"status": "started", "log": result.stdout[-2000:]}


# ------------------------------------------------------------------- UI --

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()
