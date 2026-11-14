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
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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

app = FastAPI(title="GPT-OSS-120B control panel")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


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
