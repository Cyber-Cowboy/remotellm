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
import asyncio
import json
import logging
import os
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
RATE_LIMIT_PER_MINUTE = os.environ.get("RATE_LIMIT_PER_MINUTE", "60")
ALLOWED_ORIGINS = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]

app = FastAPI(title="GPT-OSS control panel")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


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


# Runpod SKUs at the same VRAM tier are mostly interchangeable (PCIe vs.
# SXM, or just the same card sold at two price points) — Runpod's own
# capacity varies by SKU within a data center, and a deploy with only one
# GPU id can sit at zero workers forever if that exact SKU has none
# available right now (found by hand: RTX A4000 did this; so did A100
# 80GB PCIe alone, once). Passing every same-tier id as a fallback list
# costs nothing (Runpod bills whichever one it actually schedules) and
# makes that failure mode much less likely.
# A few more real Runpod enum values at the same VRAM tier *and* price as
# something in RUNPOD_GPU_CATALOG, not listed there separately, but worth
# trying as fallbacks too — same reasoning as the catalog-derived groups
# below. Keyed the same way: (vram_gb, price_hr).
_RUNPOD_EXTRA_TIER_SKUS = {
    (48, 1.22): ["NVIDIA RTX 6000 Ada Generation", "NVIDIA L40"],
}


def _runpod_tier_groups() -> list[list[str]]:
    # Grouped by (vram_gb, price_hr), not just vram_gb — A100 and H100 are
    # both "80GB" but very different prices, and silently falling back
    # from one to the other would mean paying ~2x what the cost estimate
    # shown before deploying said, without the user ever agreeing to it.
    by_tier: dict[tuple[int, float], list[str]] = {}
    for g in RUNPOD_GPU_CATALOG:
        by_tier.setdefault((g["vram_gb"], g["price_hr"]), []).append(g["id"])
    for tier, extra in _RUNPOD_EXTRA_TIER_SKUS.items():
        by_tier.setdefault(tier, []).extend(extra)
    return list(by_tier.values())


def runpod_gpu_ids_for(gpu_id: str) -> list[str]:
    """Every same-VRAM-tier SKU as a fallback list, not just the one the
    user picked. Found by hand, twice, on two different tiers (RTX A4000,
    then RTX A6000): Runpod's own capacity varies by exact SKU within a
    data center, and a deploy with only one GPU id can sit at zero
    workers forever — IN_QUEUE, no initializing/running/throttled, nothing
    — if that SKU has none available right now. The API accepts the
    request either way, so this looks identical to the Start button being
    broken. Passing every same-tier id costs nothing extra (billed at
    whichever one actually gets scheduled) and makes it much less likely."""
    for group in _runpod_tier_groups():
        if gpu_id in group:
            return group
    return [gpu_id]


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


async def _warm_runpod(endpoint_id: str, served_name: str) -> dict:
    """Runpod only spins up a worker in response to an actual job — an
    endpoint existing isn't enough. Fire a trivial job so Start actually
    starts something, instead of silently doing nothing when the endpoint
    was already there but idle/scaled-to-zero. Submission errors (bad
    endpoint, auth failure, Runpod rejecting the job) are returned, not
    just logged — a Start that silently fails is indistinguishable from a
    broken button."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"https://api.runpod.ai/v2/{endpoint_id}/run",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"},
                json={"input": {"route": "/v1/chat/completions",
                                "payload": {"model": served_name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}}},
            )
            if r.status_code >= 400:
                return {"submitted": False, "error": f"Runpod rejected the warm-up job ({r.status_code}): {r.text[:300]}"}
            return {"submitted": True, "job_id": r.json().get("id"), "error": None}
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up request to Runpod endpoint %s failed: %s", endpoint_id, e)
        return {"submitted": False, "error": f"couldn't reach Runpod to submit the warm-up job: {e}"}


async def _warm_modal(web_url: str) -> dict:
    """Same idea for Modal: an app being deployed doesn't mean a container
    is running — only real traffic to its URL triggers a cold start."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{web_url}/health")
            if r.status_code >= 400:
                return {"submitted": False, "error": f"Modal app returned {r.status_code} to the warm-up request: {r.text[:300]}"}
            return {"submitted": True, "error": None}
    except httpx.TimeoutException:
        # Expected on a real cold start — /health won't answer until the
        # container finishes booting, which can take minutes. The request
        # DID reach Modal and start a container; only a connection-level
        # failure below is a genuine submission error.
        return {"submitted": True, "error": None, "note": "still booting (timed out waiting for /health, as expected on a cold start)"}
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up request to Modal app %s failed: %s", web_url, e)
        return {"submitted": False, "error": f"couldn't reach Modal app to warm it up: {e}"}


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


STUCK_WARNING_AFTER_SECONDS = 180  # how long "0 workers since Start" is normal cold-start wait vs. likely stuck


def _stuck_warning(m: dict, active_count: int) -> str | None:
    """A GPU tier with no capacity in the data center accepts the job
    (Runpod) or the request (Modal) and then just... never schedules
    anything — no error, no running/initializing worker, forever. Found
    by hand twice (RTX A4000, then RTX A6000). There's no way to
    distinguish that from "still queued, give it a minute" from a single
    status check, so this only warns once Start has had a few minutes to
    work and nothing shows up at all."""
    if active_count > 0:
        return None
    last_attempt = m.get("last_start_attempt")
    if not last_attempt or time.time() - last_attempt < STUCK_WARNING_AFTER_SECONDS:
        return None
    return (f"Still 0 workers {int(time.time() - last_attempt)}s after Start — the chosen GPU tier "
            f"({', '.join(m.get('gpu_type_ids') or ['?'])}) may have no capacity in this data center "
            f"right now. Try removing and re-adding this model with a different GPU.")


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
            "warning": _stuck_warning(m, active),
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
            "warning": _stuck_warning(m, n),
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
            out.append({**m, "status": status, "connection": model_connection_info(m)})
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
        entry["runpod"] = await runpod_create_model(served_name, repo_id, filename, runpod_gpu_ids_for(req.gpu_id))
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
    """Ensures the endpoint/app exists, then always fires a real warm-up
    request — having an endpoint doesn't mean a GPU is running; both
    providers only scale up in response to actual traffic. Without the
    warm-up step this used to return "already running" and do nothing
    visible at all when the endpoint already existed but was scaled to
    zero, which looked exactly like the button being broken."""
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    gpu_ids = m.get("gpu_type_ids") or []
    warmup: dict = {"submitted": False, "error": "no provider-specific warm-up ran"}
    try:
        if m["provider"] == "runpod":
            if not (m.get("runpod") or {}).get("endpoint_id"):
                expanded = runpod_gpu_ids_for(gpu_ids[0]) if gpu_ids else gpu_ids
                m["runpod"] = await runpod_create_model(m["served_name"], m["repo"], m["file"], expanded)
                save_models(models)
            warmup = await _warm_runpod(m["runpod"]["endpoint_id"], m["served_name"])
        elif m["provider"] == "modal":
            if not (m.get("modal") or {}).get("app_name"):
                m["modal"] = await modal_create_model(m["served_name"], m["repo"], m["file"], gpu_ids[0])
                save_models(models)
            web_url = (m.get("modal") or {}).get("web_url")
            if web_url:
                warmup = await _warm_modal(web_url)
            else:
                warmup = {"submitted": False, "error": "model has no web_url on record — can't warm it up"}
    except HTTPException:
        raise  # from runpod_create_model/modal_create_model — already a clear error, let it surface as-is
    m["last_start_attempt"] = time.time()
    save_models(models)
    log.warning("Starting model %s: warmup=%s", model_id, warmup)
    if not warmup["submitted"]:
        # Surface this as a real error, not a 200 with a buried detail —
        # a Start that silently fails looks exactly like a broken button.
        raise HTTPException(status_code=502, detail=f"warm-up request failed: {warmup['error']}")
    return {**m, "warmup": warmup,
            "note": "warm-up request sent — cold start can take anywhere from ~1 minute to "
                    "~30 minutes depending on whether the model is already on the shared volume; "
                    "it'll show up under 'currently spending money' once a worker is actually up. "
                    "If it's still showing 0 workers after several minutes, the chosen GPU tier may "
                    "have no capacity right now — try removing and re-adding with a different one."}


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


def model_connection_info(m: dict) -> dict:
    """What a client (OpenAI SDK, AnythingLLM, curl) needs to actually talk
    to this model. Always the SAME base_url for every model on either
    provider — /v1/chat/completions etc. below dispatch by the request's
    "model" field to whichever endpoint/app that model actually lives on,
    so one gateway covers all of them instead of needing a different URL
    per model."""
    deployed = bool((m.get("runpod") or {}).get("endpoint_id") or (m.get("modal") or {}).get("web_url"))
    port = os.environ.get("DASHBOARD_PORT", "8787")
    return {
        "base_url": f"http://localhost:{port}/v1",
        "model": m["served_name"],
        "api_key": PROXY_API_KEY or None,
        "ready": deployed,
        "note": None if deployed else "Start this model first — it has no live endpoint/app to route requests to.",
    }


# --------------------------------------------------------- OpenAI gateway --
# One base_url for every model this panel knows about: the "model" field
# in the request picks which Runpod endpoint or Modal app actually serves
# it. Runpod jobs are translated the same way proxy/main.py does for a
# single model; Modal models already speak OpenAI wire format natively via
# src/modal_gateway.py, so those are a straight reverse-proxy.

def check_gateway_auth(authorization: str | None) -> None:
    if not PROXY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization[len("Bearer "):] != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def find_model_by_served_name(served_name: str) -> dict | None:
    for m in load_models():
        if m["served_name"] == served_name:
            return m
    return None


async def _runpod_job(endpoint_id: str, route: str, body: dict):
    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    stream = bool(body.get("stream", False))
    runpod_input = {"input": {"route": route, "payload": body}}

    if stream:
        async def event_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                submit = await client.post(f"{base}/run", json=runpod_input, headers=headers)
                if submit.status_code >= 400:
                    log.error("gateway: runpod /run failed %s: %s", submit.status_code, submit.text[:500])
                    yield f"data: {json.dumps({'error': 'upstream submit failed'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                job_id = submit.json()["id"]
                while True:
                    r = await client.get(f"{base}/stream/{job_id}", headers=headers)
                    if r.status_code >= 400:
                        log.error("gateway: stream poll failed %s: %s", r.status_code, r.text[:300])
                        break
                    data = r.json()
                    for item in data.get("stream", []):
                        out = item.get("output")
                        if out is not None:
                            yield f"data: {json.dumps(out)}\n\n"
                    if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                        break
                    await asyncio.sleep(0.15)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{base}/runsync", json=runpod_input, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail="upstream error")
        result = r.json()
        if result.get("status") != "COMPLETED":
            job_id = result.get("id")
            if not job_id:
                raise HTTPException(status_code=502, detail=f"unexpected upstream status: {result.get('status')}")
            while True:
                s = await client.get(f"{base}/status/{job_id}", headers=headers)
                result = s.json()
                if result.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                    break
                await asyncio.sleep(0.5)
        if result.get("status") != "COMPLETED":
            raise HTTPException(status_code=504, detail=f"job {result.get('status', 'unknown')}")
        output = result.get("output")
        output = (output[-1] if output else {}) if isinstance(output, list) else (output or {})
        if isinstance(output, dict) and "error" in output:
            raise HTTPException(status_code=502, detail=output["error"])
        return JSONResponse(content=output)


async def _modal_forward(web_url: str, route: str, body: dict):
    url = f"{web_url}{route}"
    headers = {"Content-Type": "application/json"}
    if PROXY_API_KEY:
        headers["Authorization"] = f"Bearer {PROXY_API_KEY}"
    stream = bool(body.get("stream", False))

    if stream:
        async def event_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code >= 400:
                        detail = await resp.aread()
                        log.error("gateway: modal upstream %s: %s", resp.status_code, detail[:500])
                        yield f"data: {json.dumps({'error': 'upstream error'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=body, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail="upstream error")
        return JSONResponse(content=r.json())


async def _gateway_dispatch(route: str, body: dict):
    served_name = body.get("model")
    if not served_name:
        raise HTTPException(status_code=400, detail="missing 'model' in request body")
    m = find_model_by_served_name(served_name)
    if not m:
        raise HTTPException(status_code=404, detail=f"no model named {served_name!r} is tracked by this panel")
    if m["provider"] == "runpod":
        endpoint_id = (m.get("runpod") or {}).get("endpoint_id")
        if not endpoint_id:
            raise HTTPException(status_code=409, detail=f"model {served_name!r} is stopped — Start it first")
        return await _runpod_job(endpoint_id, route, body)
    if m["provider"] == "modal":
        web_url = (m.get("modal") or {}).get("web_url")
        if not web_url:
            raise HTTPException(status_code=409, detail=f"model {served_name!r} is stopped — Start it first")
        return await _modal_forward(web_url, route, body)
    raise HTTPException(status_code=500, detail=f"unknown provider for model {served_name!r}")


@app.get("/v1/models")
async def gateway_list_models(authorization: str | None = Header(default=None)):
    check_gateway_auth(authorization)
    now = int(time.time())
    data = []
    for m in load_models():
        if (m.get("runpod") or {}).get("endpoint_id") or (m.get("modal") or {}).get("web_url"):
            data.append({"id": m["served_name"], "object": "model", "created": now, "owned_by": m["provider"]})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def gateway_chat_completions(request: Request, authorization: str | None = Header(default=None)):
    check_gateway_auth(authorization)
    body = await request.json()
    return await _gateway_dispatch("/v1/chat/completions", body)


@app.post("/v1/completions")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def gateway_completions(request: Request, authorization: str | None = Header(default=None)):
    check_gateway_auth(authorization)
    body = await request.json()
    return await _gateway_dispatch("/v1/completions", body)


# ------------------------------------------------------------------- UI --

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()
