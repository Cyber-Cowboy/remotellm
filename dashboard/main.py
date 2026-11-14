"""
Local control panel: run several models side by side, each its own
Runpod endpoint or Modal app, with a Hugging Face link, a GPU/cost picker,
and Start/Stop per model. The default view only lists models that are
currently costing money (i.e. have a GPU actually running right now).

Binds to 127.0.0.1 only by default (see scripts/run_dashboard.sh) — it
holds RUNPOD_API_KEY/MODAL_TOKEN_* server-side and every Start/Add/Stop
button here spends or stops real money, so it must never be exposed
beyond localhost without adding real authentication first.

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
import time
import urllib.parse
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dashboard")

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "deploy" / "infra_state.json"   # shared Runpod resources only (registry auth, volume)
MODELS_FILE = Path(__file__).parent / "models.json"  # every model this panel knows about
ENV_FILE = ROOT / ".env"

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DOCKERHUB_USERNAME = os.environ.get("DOCKERHUB_USERNAME", "")
RUNPOD_DATA_CENTER = os.environ.get("RUNPOD_DATA_CENTER", "US-CA-2")
RUNPOD_NETWORK_VOLUME_SIZE_GB = os.environ.get("RUNPOD_NETWORK_VOLUME_SIZE_GB", "100")
RUNPOD_IDLE_TIMEOUT_SECONDS = int(os.environ.get("MODEL_IDLE_TIMEOUT_SECONDS", "60"))

app = FastAPI(title="GPT-OSS control panel")


# ------------------------------------------------------------- storage --

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def update_env_file(updates: dict[str, str]) -> None:
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


def load_models() -> list[dict]:
    if not MODELS_FILE.exists():
        _migrate_legacy_model()
    if MODELS_FILE.exists():
        return json.loads(MODELS_FILE.read_text())
    return []


def save_models(models: list[dict]) -> None:
    MODELS_FILE.write_text(json.dumps(models, indent=2))


def _migrate_legacy_model() -> None:
    """One-time: the very first deployment this project made (before this
    panel supported more than one model) tracked its endpoint/template in
    deploy/infra_state.json directly. Pull that into models.json as model
    #1 and strip those two keys back out of infra_state.json, which from
    here on only holds resources *shared* across every model (the registry
    auth and the network volume)."""
    state = load_state()
    if not (state.get("endpoint_id") and state.get("template_id")):
        return
    served_name = os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-uncensored")
    gpu_ids = [g for g in os.environ.get(
        "RUNPOD_GPU_TYPE_IDS",
        "NVIDIA A100 80GB PCIe,NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3",
    ).split(",") if g]
    entry = {
        "id": f"runpod-{served_name}",
        "served_name": served_name,
        "provider": "runpod",
        "repo": os.environ.get("MODEL_REPO", ""),
        "file": os.environ.get("MODEL_FILE", ""),
        "gpu_type_ids": gpu_ids,
        "runpod": {"endpoint_id": state["endpoint_id"], "template_id": state["template_id"]},
        "created_at": time.time(),
    }
    save_models([entry])
    state.pop("endpoint_id", None)
    state.pop("template_id", None)
    save_state(state)
    log.warning("Migrated legacy single-model deployment into models.json as %s", entry["id"])


# ------------------------------------------------------- Hugging Face --

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


async def list_repo_files(repo_id: str) -> list[dict]:
    """The tree API (not /api/models/{repo}) specifically because it's the
    one that includes file sizes — needed for the GPU/VRAM/cost estimate."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(f"https://huggingface.co/api/models/{repo_id}/tree/main")
        if r.status_code == 404:
            raise ValueError(f"repo not found on Hugging Face: {repo_id}")
        r.raise_for_status()
        return r.json()


async def find_gguf_file(repo_id: str) -> tuple[str, int | None]:
    files = await list_repo_files(repo_id)
    ggufs = [f for f in files if f.get("path", "").lower().endswith(".gguf")]
    if len(ggufs) == 1:
        return ggufs[0]["path"], ggufs[0].get("size")
    if not ggufs:
        raise ValueError(f"no .gguf files found in {repo_id} (only GGUF models are supported here)")
    raise ValueError(
        f"{len(ggufs)} .gguf files found in {repo_id} — paste a direct link to the one you want, e.g. "
        f"https://huggingface.co/{repo_id}/resolve/main/{ggufs[0]['path']}"
    )


async def file_size(repo_id: str, filename: str) -> int | None:
    try:
        files = await list_repo_files(repo_id)
    except ValueError:
        return None
    for f in files:
        if f.get("path") == filename:
            return f.get("size")
    return None


def derive_served_name(repo_id: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", repo_id.split("/")[-1].lower()).strip("-")
    return name or "custom-model"


async def resolve_hf_model(hf_url: str) -> tuple[str, str, int | None]:
    """-> (repo_id, filename, size_bytes). Raises HTTPException on anything
    that should stop the flow (ambiguous repo, missing file, etc.)."""
    try:
        repo_id, filename = parse_hf_input(hf_url)
        if filename is None:
            filename, size_bytes = await find_gguf_file(repo_id)
        elif not filename.lower().endswith(".gguf"):
            raise ValueError(f"only .gguf files are supported by llama-server here, got: {filename!r}")
        else:
            size_bytes = await file_size(repo_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resolve_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.head(resolve_url)
        if r.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"file not found on Hugging Face ({r.status_code}): {resolve_url}")
        if size_bytes is None:
            cl = r.headers.get("content-length")
            size_bytes = int(cl) if cl else None
    return repo_id, filename, size_bytes


# ----------------------------------------------------- GPU / cost data --
# Real per-GPU prices, hand-checked against each provider's own pricing
# page (Runpod Serverless; Modal's per-second rates × 3600) — update these
# if/when the providers change them, they're not fetched live.

RUNPOD_GPU_CATALOG = [
    {"id": "NVIDIA RTX A4000", "vram_gb": 16, "price_hr": 0.58},
    {"id": "NVIDIA L4", "vram_gb": 24, "price_hr": 0.75},
    {"id": "NVIDIA RTX A6000", "vram_gb": 48, "price_hr": 1.22},
    {"id": "NVIDIA A40", "vram_gb": 48, "price_hr": 1.22},
    {"id": "NVIDIA L40S", "vram_gb": 48, "price_hr": 1.89},
    {"id": "NVIDIA A100 80GB PCIe", "vram_gb": 80, "price_hr": 2.99},
    {"id": "NVIDIA A100-SXM4-80GB", "vram_gb": 80, "price_hr": 2.99},
    {"id": "NVIDIA H100 PCIe", "vram_gb": 80, "price_hr": 5.89},
    {"id": "NVIDIA H100 80GB HBM3", "vram_gb": 80, "price_hr": 5.89},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "vram_gb": 96, "price_hr": 3.69},
    {"id": "NVIDIA H200", "vram_gb": 141, "price_hr": 6.99},
    {"id": "NVIDIA B200", "vram_gb": 180, "price_hr": 10.79},
]

MODAL_GPU_CATALOG = [
    {"id": "T4", "vram_gb": 16, "price_hr": round(0.000164 * 3600, 3)},
    {"id": "L4", "vram_gb": 24, "price_hr": round(0.000222 * 3600, 3)},
    {"id": "A10", "vram_gb": 24, "price_hr": round(0.000306 * 3600, 3)},
    {"id": "L40S", "vram_gb": 48, "price_hr": round(0.000542 * 3600, 3)},
    {"id": "A100-40GB", "vram_gb": 40, "price_hr": round(0.000583 * 3600, 3)},
    {"id": "A100-80GB", "vram_gb": 80, "price_hr": round(0.000694 * 3600, 3)},
    {"id": "RTX PRO 6000", "vram_gb": 96, "price_hr": round(0.000842 * 3600, 3)},
    {"id": "H100", "vram_gb": 80, "price_hr": round(0.001097 * 3600, 3)},
    {"id": "H200", "vram_gb": 141, "price_hr": round(0.001261 * 3600, 3)},
    {"id": "B200", "vram_gb": 180, "price_hr": round(0.001736 * 3600, 3)},
]

CATALOGS = {"runpod": RUNPOD_GPU_CATALOG, "modal": MODAL_GPU_CATALOG}


def gpu_price(provider: str, gpu_id: str) -> float | None:
    for g in CATALOGS.get(provider, []):
        if g["id"] == gpu_id:
            return g["price_hr"]
    return None


class ResolveRequest(BaseModel):
    hf_url: str
    provider: str


@app.post("/api/models/resolve")
async def models_resolve(req: ResolveRequest):
    catalog = CATALOGS.get(req.provider)
    if not catalog:
        raise HTTPException(status_code=400, detail=f"unknown provider: {req.provider!r}")

    repo_id, filename, size_bytes = await resolve_hf_model(req.hf_url)
    file_gb = size_bytes / 1e9 if size_bytes else None
    # weights + ~15% for KV cache/runtime overhead + a small flat buffer
    required_gb = file_gb * 1.15 + 2 if file_gb else None

    options = []
    recommended_id = None
    for g in catalog:
        fits = required_gb is not None and required_gb <= g["vram_gb"]
        options.append({**g, "fits": fits})
        if fits and recommended_id is None:
            recommended_id = g["id"]
    fits_any = required_gb is None or required_gb <= catalog[-1]["vram_gb"]
    if recommended_id is None:
        recommended_id = catalog[-1]["id"]  # nothing fits — still pick *something*, flagged by fits_any

    return {
        "repo": repo_id,
        "file": filename,
        "served_name": derive_served_name(repo_id),
        "file_gb": round(file_gb, 1) if file_gb else None,
        "estimated_vram_gb": round(required_gb, 1) if required_gb else None,
        "options": options,
        "recommended_id": recommended_id,
        "fits_any": fits_any,
    }


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


async def runpod_create_model(served_name: str, repo: str, filename: str, gpu_ids: list[str]) -> dict:
    state = load_state()
    network_volume_id = state.get("network_volume_id")
    registry_auth_id = state.get("registry_auth_id")
    if not network_volume_id:
        raise HTTPException(
            status_code=400,
            detail="no shared Runpod network volume yet — run `PROVIDER=runpod bash deploy.sh` once first",
        )
    if not DOCKERHUB_USERNAME:
        raise HTTPException(status_code=400, detail="DOCKERHUB_USERNAME not set")

    env = {
        "SERVED_MODEL_NAME": served_name,
        "MODEL_REPO": repo,
        "MODEL_FILE": filename,
        "MODEL_DIR": f"/runpod-volume/models/{served_name}",
        "MAX_MODEL_LEN": os.environ.get("MAX_MODEL_LEN", "8192"),
        "N_GPU_LAYERS": os.environ.get("N_GPU_LAYERS", "999"),
        "N_CPU_MOE": os.environ.get("N_CPU_MOE", "0"),
        "MAX_CONCURRENT": os.environ.get("MAX_CONCURRENT", "2"),
    }
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        tmpl_body = {
            "name": f"gptoss-{served_name}-template",
            "imageName": f"docker.io/{DOCKERHUB_USERNAME}/gptoss-120b-worker:latest",
            "category": "NVIDIA", "isServerless": True, "containerDiskInGb": 20,
            "ports": ["8080/http"], "env": env,
        }
        if registry_auth_id:
            tmpl_body["containerRegistryAuthId"] = registry_auth_id
        tmpl = await client.post("https://rest.runpod.io/v1/templates", headers=headers, json=tmpl_body)
        if tmpl.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"template create failed: {tmpl.text}")
        template_id = tmpl.json()["id"]

        ep_body = {
            "name": f"gptoss-{served_name}",
            "templateId": template_id,
            "computeType": "GPU",
            "gpuTypeIds": gpu_ids,
            "gpuCount": 1,
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": RUNPOD_IDLE_TIMEOUT_SECONDS,
            "flashboot": True,
            "networkVolumeId": network_volume_id,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 1,
        }
        ep = await client.post("https://rest.runpod.io/v1/endpoints", headers=headers, json=ep_body)
        if ep.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"endpoint create failed: {ep.text}")
        endpoint_id = ep.json()["id"]
    return {"template_id": template_id, "endpoint_id": endpoint_id}


async def runpod_delete_model(m: dict) -> None:
    rp = m.get("runpod") or {}
    async with httpx.AsyncClient(timeout=30) as client:
        for key, path in (("endpoint_id", "endpoints"), ("template_id", "templates")):
            rid = rp.get(key)
            if rid:
                await client.delete(
                    f"https://rest.runpod.io/v1/{path}/{rid}",
                    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                )


@app.get("/api/runpod/status")
async def runpod_status():
    account = await runpod_account()
    state = load_state()
    return {
        "configured": bool(RUNPOD_API_KEY),
        "account": account,
        "has_shared_volume": bool(state.get("network_volume_id")),
    }


@app.post("/api/runpod/teardown")
async def runpod_teardown():
    """Delete every paid Runpod resource across every model: each model's
    endpoint+template, then the shared network volume (which holds every
    model's downloaded weights — the next deploy re-downloads from
    scratch). Leaves nothing billing at all, not even storage."""
    models = load_models()
    deleted_models = []
    for m in models:
        if m["provider"] == "runpod":
            await runpod_delete_model(m)
            deleted_models.append(m["id"])
    state = load_state()
    volume_id = state.get("network_volume_id")
    if volume_id:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.delete(
                f"https://rest.runpod.io/v1/networkvolumes/{volume_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
            if r.status_code not in (200, 204, 404):
                raise HTTPException(status_code=502, detail=f"volume delete failed: {r.text}")
    save_state({})
    remaining = [m for m in models if m["provider"] != "runpod"]
    save_models(remaining)
    log.warning("Runpod full teardown: removed models %s + shared volume", deleted_models)
    return {"status": "deleted", "removed_models": deleted_models, "volume_deleted": bool(volume_id)}


# ------------------------------------------------------------------ Modal --

def modal_env() -> dict:
    return dict(os.environ)


async def modal_create_model(served_name: str, repo: str, filename: str, gpu_id: str) -> dict:
    secret_name = f"gptoss-worker-secrets-{served_name}"
    cmd = [
        "python3", "-m", "modal", "secret", "create", secret_name,
        f"SERVED_MODEL_NAME={served_name}",
        f"MODEL_REPO={repo}",
        f"MODEL_FILE={filename}",
        f"MODEL_DIR=/cache/models/{served_name}",
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
    r1 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60)
    if r1.returncode != 0:
        raise HTTPException(status_code=502, detail=f"secret create failed: {(r1.stderr or r1.stdout)[-2000:]}")

    deploy_env = modal_env()
    deploy_env["MODAL_APP_SUFFIX"] = f"-{served_name}"
    deploy_env["MODAL_WORKER_SECRET_NAME"] = secret_name
    deploy_env["MODAL_GPU_TYPES"] = gpu_id
    r2 = subprocess.run(
        ["python3", "-m", "modal", "deploy", str(ROOT / "modal_app.py")],
        cwd=ROOT, capture_output=True, text=True, env=deploy_env, timeout=600,
    )
    if r2.returncode != 0:
        raise HTTPException(status_code=502, detail=f"modal deploy failed: {(r2.stderr or r2.stdout)[-3000:]}")

    url_match = re.search(r"https://[A-Za-z0-9.-]*modal\.run[A-Za-z0-9/_-]*", r2.stdout)
    return {
        "app_name": f"gptoss-120b-{served_name}",
        "secret_name": secret_name,
        "web_url": url_match.group(0) if url_match else None,
    }


async def modal_delete_model(m: dict) -> None:
    info = m.get("modal") or {}
    if info.get("app_name"):
        subprocess.run(["python3", "-m", "modal", "app", "stop", info["app_name"], "-y"],
                        cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60)
    if info.get("secret_name"):
        subprocess.run(["python3", "-m", "modal", "secret", "delete", info["secret_name"], "-y"],
                        cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60)


async def modal_running_containers(app_name: str) -> int:
    try:
        result = subprocess.run(
            ["python3", "-m", "modal", "container", "list", "--json"],
            cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=30,
        )
    except Exception:  # noqa: BLE001
        return 0
    if result.returncode != 0 or not result.stdout.strip():
        return 0
    try:
        containers = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0
    # Best-effort: Modal's container-list JSON key casing for the app name
    # wasn't verified against a real running container (the Modal deploy
    # path is blocked on their payment-method requirement — see README) —
    # this matches on the app name appearing anywhere in each row rather
    # than a specific key, which is slightly loose but tolerant of that.
    return sum(1 for c in containers if app_name in json.dumps(c))


@app.get("/api/modal/status")
async def modal_status():
    return {"configured": bool(os.environ.get("MODAL_TOKEN_ID"))}


@app.post("/api/modal/teardown")
async def modal_teardown():
    """Stop every Modal model's app + its secret, then the shared model
    volume and the shared registry secret. Leaves nothing behind."""
    models = load_models()
    removed = []
    for m in models:
        if m["provider"] == "modal":
            await modal_delete_model(m)
            removed.append(m["id"])
    subprocess.run(
        ["python3", "-m", "modal", "volume", "delete", os.environ.get("MODAL_VOLUME_NAME", "gptoss-120b-model-cache"), "-y"],
        cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60,
    )
    subprocess.run(
        ["python3", "-m", "modal", "secret", "delete", "dockerhub-registry-secret", "-y"],
        cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60,
    )
    remaining = [m for m in models if m["provider"] != "modal"]
    save_models(remaining)
    log.warning("Modal full teardown: removed models %s + shared volume/secret", removed)
    return {"status": "deleted", "removed_models": removed}


# ----------------------------------------------------------------- models --

class AddModelRequest(BaseModel):
    hf_url: str
    provider: str
    gpu_id: str
    served_name: str | None = None


async def model_live_status(m: dict) -> dict:
    if m["provider"] == "runpod":
        endpoint_id = (m.get("runpod") or {}).get("endpoint_id")
        if not endpoint_id:
            return {"spending": False, "state": "stopped", "est_cost_per_hr": 0}
        workers = await runpod_workers(endpoint_id)
        if not workers:
            return {"spending": False, "state": "stopped", "est_cost_per_hr": 0}
        summary = workers.get("summary", {})
        active = summary.get("running", 0) + summary.get("initializing", 0) + summary.get("throttled", 0)
        price = gpu_price("runpod", (m.get("gpu_type_ids") or [None])[0])
        return {
            "spending": active > 0,
            "state": "running" if active > 0 else "idle",
            "workers": summary,
            "est_cost_per_hr": round(price * active, 2) if price else 0,
        }
    if m["provider"] == "modal":
        app_name = (m.get("modal") or {}).get("app_name")
        if not app_name:
            return {"spending": False, "state": "stopped", "est_cost_per_hr": 0}
        n = await modal_running_containers(app_name)
        price = gpu_price("modal", (m.get("gpu_type_ids") or [None])[0])
        return {
            "spending": n > 0,
            "state": "running" if n > 0 else "stopped",
            "containers": n,
            "est_cost_per_hr": round(price * n, 2) if price else 0,
        }
    return {"spending": False, "state": "unknown", "est_cost_per_hr": 0}


@app.get("/api/models")
async def models_list(all: bool = False):
    models = load_models()
    out = []
    for m in models:
        status = await model_live_status(m)
        if all or status["spending"]:
            out.append({**m, "status": status})
    return {"models": out, "total_tracked": len(models)}


@app.post("/api/models")
async def models_add(req: AddModelRequest):
    if req.provider not in CATALOGS:
        raise HTTPException(status_code=400, detail=f"unknown provider: {req.provider!r}")
    if gpu_price(req.provider, req.gpu_id) is None:
        raise HTTPException(status_code=400, detail=f"unknown GPU id for {req.provider}: {req.gpu_id!r}")

    repo_id, filename, _ = await resolve_hf_model(req.hf_url)
    served_name = (req.served_name or "").strip() or derive_served_name(repo_id)

    models = load_models()
    if any(m["served_name"] == served_name and m["provider"] == req.provider for m in models):
        raise HTTPException(status_code=400, detail=f"a {req.provider} model named {served_name!r} already exists")

    entry = {
        "id": f"{req.provider}-{served_name}",
        "served_name": served_name,
        "provider": req.provider,
        "repo": repo_id,
        "file": filename,
        "gpu_type_ids": [req.gpu_id],
        "created_at": time.time(),
    }
    if req.provider == "runpod":
        entry["runpod"] = await runpod_create_model(served_name, repo_id, filename, [req.gpu_id])
    else:
        entry["modal"] = await modal_create_model(served_name, repo_id, filename, req.gpu_id)

    models.append(entry)
    save_models(models)
    log.warning("Added model %s on %s (%s)", served_name, req.provider, req.gpu_id)
    return entry


@app.post("/api/models/{model_id}/stop")
async def models_stop(model_id: str):
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    if m["provider"] == "runpod":
        await runpod_delete_model(m)
        m["runpod"] = {}
    elif m["provider"] == "modal":
        info = m.get("modal") or {}
        if info.get("app_name"):
            subprocess.run(["python3", "-m", "modal", "app", "stop", info["app_name"], "-y"],
                            cwd=ROOT, capture_output=True, text=True, env=modal_env(), timeout=60)
        m["modal"] = {**info, "app_name": None}
    save_models(models)
    log.warning("Stopped model %s", model_id)
    return {"status": "stopped"}


@app.post("/api/models/{model_id}/start")
async def models_start(model_id: str):
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    gpu_ids = m.get("gpu_type_ids") or []
    if m["provider"] == "runpod":
        if (m.get("runpod") or {}).get("endpoint_id"):
            return {"status": "already running"}
        m["runpod"] = await runpod_create_model(m["served_name"], m["repo"], m["file"], gpu_ids)
    elif m["provider"] == "modal":
        if (m.get("modal") or {}).get("app_name"):
            return {"status": "already running"}
        m["modal"] = await modal_create_model(m["served_name"], m["repo"], m["file"], gpu_ids[0])
    save_models(models)
    log.warning("Started model %s", model_id)
    return m


@app.delete("/api/models/{model_id}")
async def models_remove(model_id: str):
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    if m["provider"] == "runpod":
        await runpod_delete_model(m)
    elif m["provider"] == "modal":
        await modal_delete_model(m)
    models = [x for x in models if x["id"] != model_id]
    save_models(models)
    log.warning("Removed model %s", model_id)
    return {"status": "removed"}


# ------------------------------------------------------------------- UI --

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()
