"""
Local control panel: run several models side by side on Runpod, each its
own endpoint, with a Hugging Face link, a GPU/cost picker, and Start/Stop
per model. The default view only lists models that are currently costing
money (i.e. have a GPU actually running right now).

Binds to 127.0.0.1 only by default (see scripts/run_dashboard.sh) — it
holds RUNPOD_API_KEY server-side and every Start/Add/Stop button here
spends or stops real money, so it must never be exposed beyond localhost
without adding real authentication first.

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
from contextlib import asynccontextmanager
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

AUTO_STOP_CHECK_INTERVAL_SECONDS = 60

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_auto_stop_loop())
    yield
    task.cancel()


app = FastAPI(title="GPT-OSS control panel", lifespan=lifespan)
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


async def fetch_context_recommendation(repo_id: str) -> tuple[int | None, int | None]:
    """-> (recommended_context_window, recommended_max_tokens). GGUF-only
    repos (the normal case here) usually don't publish a config.json —
    the base model's config lives in a different repo entirely — but HF's
    own model-info API reads it straight out of the GGUF file's own
    metadata when asked to (?expand[]=gguf), which is always present and
    authoritative regardless of whether this particular repo has a
    config.json. max_tokens is a client-side generation-length choice,
    not a model property, so there's no "correct" value to read — this
    suggests half the context window (leaving room for the prompt),
    capped to a sane range, as a reasonable default rather than nothing
    at all."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(f"https://huggingface.co/api/models/{repo_id}", params={"expand[]": "gguf"})
            if r.status_code >= 400:
                return None, None
            ctx = r.json().get("gguf", {}).get("context_length")
    except Exception as e:  # noqa: BLE001 — best-effort; absence of a recommendation isn't an error
        log.warning("couldn't fetch GGUF metadata for %s: %s", repo_id, e)
        return None, None
    if not isinstance(ctx, int):
        return None, None
    max_tokens = max(256, min(ctx // 2, 8192))
    return ctx, max_tokens


# ----------------------------------------------------- GPU / cost data --
# Real per-GPU prices, hand-checked against Runpod's own Serverless
# pricing page — update this if/when Runpod changes them, it's not
# fetched live.

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


def gpu_price(gpu_id: str) -> float | None:
    for g in RUNPOD_GPU_CATALOG:
        if g["id"] == gpu_id:
            return g["price_hr"]
    return None


def gpu_options_for(required_gb: float | None) -> dict:
    options = []
    recommended_id = None
    for g in RUNPOD_GPU_CATALOG:
        fits = required_gb is not None and required_gb <= g["vram_gb"]
        options.append({**g, "fits": fits})
        if fits and recommended_id is None:
            recommended_id = g["id"]
    fits_any = required_gb is None or required_gb <= RUNPOD_GPU_CATALOG[-1]["vram_gb"]
    if recommended_id is None:
        recommended_id = RUNPOD_GPU_CATALOG[-1]["id"]  # nothing fits — still pick *something*, flagged by fits_any
    return {"options": options, "recommended_id": recommended_id, "fits_any": fits_any}


async def gpu_options_with_stock(required_gb: float | None) -> dict:
    result = gpu_options_for(required_gb)
    stock = await runpod_gpu_stock()
    for opt in result["options"]:
        opt["stock_status"] = stock.get(opt["id"])
    return result


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


@app.post("/api/models/resolve")
async def models_resolve(req: ResolveRequest):
    repo_id, filename, size_bytes = await resolve_hf_model(req.hf_url)
    file_gb = size_bytes / 1e9 if size_bytes else None
    # weights + ~15% for KV cache/runtime overhead + a small flat buffer
    required_gb = file_gb * 1.15 + 2 if file_gb else None
    ctx, max_tokens = await fetch_context_recommendation(repo_id)

    return {
        "repo": repo_id,
        "file": filename,
        "served_name": derive_served_name(repo_id),
        "file_gb": round(file_gb, 1) if file_gb else None,
        "estimated_vram_gb": round(required_gb, 1) if required_gb else None,
        "recommended_context_window": ctx,
        "recommended_max_tokens": max_tokens,
        **(await gpu_options_with_stock(required_gb)),
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


_gpu_stock_cache: dict = {"at": 0.0, "data": {}}
GPU_STOCK_CACHE_SECONDS = 30


async def runpod_gpu_stock() -> dict[str, str | None]:
    """Runpod's own "High"/"Medium"/"Low" stock signal per GPU id, from the
    same GraphQL field used to diagnose real capacity shortages during
    development (see README "Known limitations"). There's no working
    per-data-center breakdown — passing dataCenterId to lowestPrice makes
    stockStatus come back null instead of filtered, confirmed by hand — so
    this is Runpod's global figure, the best available proxy for "will
    this GPU actually start", not specific to RUNPOD_DATA_CENTER."""
    now = time.time()
    if now - _gpu_stock_cache["at"] < GPU_STOCK_CACHE_SECONDS and _gpu_stock_cache["data"]:
        return _gpu_stock_cache["data"]
    if not RUNPOD_API_KEY:
        return {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.runpod.io/graphql",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json={"query": "query { gpuTypes { id lowestPrice(input:{gpuCount:1}){ stockStatus } } }"},
            )
            r.raise_for_status()
            rows = r.json()["data"]["gpuTypes"]
    except Exception as e:
        log.warning("couldn't fetch GPU stock status: %s", e)
        return _gpu_stock_cache["data"]
    data = {row["id"]: (row.get("lowestPrice") or {}).get("stockStatus") for row in rows}
    _gpu_stock_cache["at"] = now
    _gpu_stock_cache["data"] = data
    return data


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


async def runpod_create_model(served_name: str, repo: str, filename: str, gpu_ids: list[str],
                               context_window: int | None = None) -> dict:
    state = load_state()
    network_volume_id = state.get("network_volume_id")
    registry_auth_id = state.get("registry_auth_id")
    if not network_volume_id:
        raise HTTPException(
            status_code=400,
            detail="no shared Runpod network volume yet — run `bash deploy.sh` once first",
        )
    if not DOCKERHUB_USERNAME:
        raise HTTPException(status_code=400, detail="DOCKERHUB_USERNAME not set")

    env = {
        "SERVED_MODEL_NAME": served_name,
        "MODEL_REPO": repo,
        "MODEL_FILE": filename,
        "MODEL_DIR": f"/runpod-volume/models/{served_name}",
        "MAX_MODEL_LEN": str(context_window or os.environ.get("MAX_MODEL_LEN", "8192")),
        "N_GPU_LAYERS": os.environ.get("N_GPU_LAYERS", "999"),
        "N_CPU_MOE": os.environ.get("N_CPU_MOE", "0"),
        "MAX_CONCURRENT": os.environ.get("MAX_CONCURRENT", "2"),
    }
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        tmpl_body = {
            # Runpod holds a deleted endpoint's template "pending cleanup"
            # for up to ~90 minutes before its name is free again — a
            # purely served_name-derived name collides on the very next
            # Stop -> Start within that window ("Template name must be
            # unique"). A short suffix sidesteps it rather than hoping
            # the timing never lines up.
            "name": f"gptoss-{served_name}-template-{int(time.time())}",
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
            "name": f"gptoss-{served_name}-{int(time.time())}",
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


async def runpod_retarget_gpu(template_id: str, endpoint_id: str, gpu_ids: list[str]) -> None:
    """Switch an existing endpoint to a different GPU tier in place — no
    need to delete/recreate, which would briefly drop the model off the
    "tracked" list for no reason."""
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.patch(
            f"https://rest.runpod.io/v1/endpoints/{endpoint_id}",
            headers=headers, json={"gpuTypeIds": gpu_ids},
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"GPU retarget failed: {r.text}")


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


async def _check_auto_stop() -> None:
    """Per-model auto-stop: deletes that model's endpoint+template (same
    as a manual Stop) once `auto_stop_minutes` have passed since its last
    request — independent of Runpod's own worker-level idleTimeout, which
    only scales the GPU to zero and leaves the endpoint itself sitting
    there. Measured from the last real request if there's been one, else
    from when it was last started, so a model nobody ever used doesn't
    linger forever but also isn't killed before anyone got to try it."""
    models = load_models()
    changed = False
    for m in models:
        minutes = m.get("auto_stop_minutes")
        if not minutes:
            continue
        if not (m.get("runpod") or {}).get("endpoint_id"):
            continue
        reference = m.get("last_request_at") or m.get("last_start_attempt") or m.get("created_at")
        if not reference or time.time() - reference < minutes * 60:
            continue
        log.warning("Auto-stopping model %s after %s min with no requests", m["id"], minutes)
        await runpod_delete_model(m)
        m["runpod"] = {}
        changed = True
    if changed:
        save_models(models)


async def _auto_stop_loop() -> None:
    while True:
        try:
            await _check_auto_stop()
        except Exception as e:  # noqa: BLE001 — one bad check must not kill the loop
            log.warning("auto-stop check failed: %s", e)
        await asyncio.sleep(AUTO_STOP_CHECK_INTERVAL_SECONDS)


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
    save_models([])
    log.warning("Runpod full teardown: removed models %s + shared volume", deleted_models)
    return {"status": "deleted", "removed_models": deleted_models, "volume_deleted": bool(volume_id)}


# ----------------------------------------------------------------- models --

class AddModelRequest(BaseModel):
    hf_url: str
    gpu_id: str
    served_name: str | None = None
    context_window: int | None = None
    auto_stop_minutes: int | None = None


class StartModelRequest(BaseModel):
    gpu_id: str | None = None


class AutoStopRequest(BaseModel):
    minutes: int | None = None  # null/0 disables it


STUCK_WARNING_AFTER_SECONDS = 180  # how long "0 workers since Start" is normal cold-start wait vs. likely stuck


def _stuck_warning(m: dict, active_count: int) -> str | None:
    """A GPU tier with no capacity in the data center accepts the job and
    then just... never schedules anything — no error, no
    initializing/running worker, forever. Found by hand twice (RTX A4000,
    then RTX A6000). There's no way to distinguish that from "still
    queued, give it a minute" from a single status check, so this only
    warns once Start has had a few minutes to work and nothing shows up
    at all."""
    if active_count > 0:
        return None
    last_attempt = m.get("last_start_attempt")
    if not last_attempt or time.time() - last_attempt < STUCK_WARNING_AFTER_SECONDS:
        return None
    return (f"Still 0 workers {int(time.time() - last_attempt)}s after Start — the chosen GPU tier "
            f"({', '.join(m.get('gpu_type_ids') or ['?'])}) may have no capacity in this data center "
            f"right now. Try Start again with a different GPU.")


async def model_live_status(m: dict) -> dict:
    endpoint_id = (m.get("runpod") or {}).get("endpoint_id")
    if not endpoint_id:
        return {"spending": False, "state": "stopped", "est_cost_per_hr": 0}
    workers = await runpod_workers(endpoint_id)
    if not workers:
        return {"spending": False, "state": "stopped", "est_cost_per_hr": 0}
    summary = workers.get("summary", {})
    active = summary.get("running", 0) + summary.get("initializing", 0) + summary.get("throttled", 0)
    price = gpu_price((m.get("gpu_type_ids") or [None])[0])
    last_attempt = m.get("last_start_attempt")
    # No worker yet, but Start was pressed recently enough that this is
    # still the normal cold-start wait, not the "likely stuck" case
    # _stuck_warning would flag — show that as its own "starting" state
    # rather than lumping it in with "idle" (which, to someone who just
    # clicked Start, looks exactly like nothing happened).
    starting = active == 0 and last_attempt and (time.time() - last_attempt) < STUCK_WARNING_AFTER_SECONDS
    return {
        "spending": active > 0,
        "state": "running" if active > 0 else ("starting" if starting else "idle"),
        "workers": summary,
        "est_cost_per_hr": round(price * active, 2) if price else 0,
        "seconds_since_start": int(time.time() - last_attempt) if last_attempt else None,
        "warning": _stuck_warning(m, active),
    }


async def _ensure_context_recommendation(m: dict, models: list[dict]) -> None:
    """Backfill recommended_context_window/recommended_max_tokens for
    models added before this existed, once, the first time they're
    listed — not on every poll."""
    if "recommended_context_window" in m:
        return
    ctx, max_tokens = await fetch_context_recommendation(m["repo"])
    m["recommended_context_window"] = ctx
    m["recommended_max_tokens"] = max_tokens
    save_models(models)


@app.get("/api/models")
async def models_list(all: bool = False):
    models = load_models()
    out = []
    for m in models:
        await _ensure_context_recommendation(m, models)
        status = await model_live_status(m)
        if all or status["spending"] or status["state"] == "starting" or status.get("warning"):
            out.append({**m, "status": status, "connection": model_connection_info(m)})
    return {"models": out, "total_tracked": len(models)}


@app.get("/api/models/{model_id}/gpu-options")
async def models_gpu_options(model_id: str):
    """What to show a GPU picker when (re)starting this model: every
    Runpod tier with its real $/hr, and which one(s) actually fit based on
    the model's real file size (recomputed from the stored repo/file if
    not already cached on the model)."""
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    required_gb = m.get("estimated_vram_gb")
    if required_gb is None:
        size_bytes = await file_size(m["repo"], m["file"])
        if size_bytes:
            required_gb = round(size_bytes / 1e9 * 1.15 + 2, 1)
            m["estimated_vram_gb"] = required_gb
            save_models(models)
    return {"estimated_vram_gb": required_gb, **(await gpu_options_with_stock(required_gb))}


@app.post("/api/models")
async def models_add(req: AddModelRequest):
    if gpu_price(req.gpu_id) is None:
        raise HTTPException(status_code=400, detail=f"unknown Runpod GPU id: {req.gpu_id!r}")

    repo_id, filename, size_bytes = await resolve_hf_model(req.hf_url)
    served_name = (req.served_name or "").strip() or derive_served_name(repo_id)

    models = load_models()
    if any(m["served_name"] == served_name for m in models):
        raise HTTPException(status_code=400, detail=f"a model named {served_name!r} already exists")

    ctx, default_max_tokens = await fetch_context_recommendation(repo_id)
    context_window = req.context_window or ctx

    entry = {
        "id": f"runpod-{served_name}",
        "served_name": served_name,
        "provider": "runpod",
        "repo": repo_id,
        "file": filename,
        "gpu_type_ids": [req.gpu_id],
        "estimated_vram_gb": round(size_bytes / 1e9 * 1.15 + 2, 1) if size_bytes else None,
        "recommended_context_window": ctx,
        "recommended_max_tokens": default_max_tokens,
        "auto_stop_minutes": req.auto_stop_minutes,
        "created_at": time.time(),
    }
    entry["runpod"] = await runpod_create_model(served_name, repo_id, filename,
                                                  runpod_gpu_ids_for(req.gpu_id), context_window)

    models.append(entry)
    save_models(models)
    log.warning("Added model %s on runpod (%s)", served_name, req.gpu_id)
    return entry


@app.post("/api/models/{model_id}/stop")
async def models_stop(model_id: str):
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    await runpod_delete_model(m)
    m["runpod"] = {}
    save_models(models)
    log.warning("Stopped model %s", model_id)
    return {"status": "stopped"}


@app.post("/api/models/{model_id}/start")
async def models_start(model_id: str, req: StartModelRequest | None = None):
    """If the model's endpoint doesn't exist (first start, or after Stop),
    a gpu_id is required — re-creating always means choosing hardware,
    with its price, rather than silently reusing whatever was picked
    last time. If the endpoint already exists, gpu_id is optional: pass
    it to retarget to a different tier in place, or omit it to just wake
    the existing one back up."""
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    gpu_id = req.gpu_id if req else None
    rp = m.get("runpod") or {}

    if not rp.get("endpoint_id"):
        if not gpu_id:
            raise HTTPException(
                status_code=400,
                detail="this model has no live endpoint — pick a GPU to (re)create it (see "
                       "/api/models/{id}/gpu-options)",
            )
        m["gpu_type_ids"] = [gpu_id]
        m["runpod"] = await runpod_create_model(
            m["served_name"], m["repo"], m["file"], runpod_gpu_ids_for(gpu_id), m.get("recommended_context_window"),
        )
        save_models(models)
    elif gpu_id and gpu_id != (m.get("gpu_type_ids") or [None])[0]:
        await runpod_retarget_gpu(m["runpod"]["template_id"], m["runpod"]["endpoint_id"], runpod_gpu_ids_for(gpu_id))
        m["gpu_type_ids"] = [gpu_id]
        save_models(models)

    warmup = await _warm_runpod(m["runpod"]["endpoint_id"], m["served_name"])
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
                    "have no capacity right now — try Start again with a different GPU."}


@app.post("/api/models/{model_id}/auto-stop")
async def models_set_auto_stop(model_id: str, req: AutoStopRequest):
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    m["auto_stop_minutes"] = req.minutes or None
    save_models(models)
    log.warning("Set auto_stop_minutes=%s for model %s", m["auto_stop_minutes"], model_id)
    return {"auto_stop_minutes": m["auto_stop_minutes"]}


@app.delete("/api/models/{model_id}")
async def models_remove(model_id: str):
    models = load_models()
    m = next((x for x in models if x["id"] == model_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    await runpod_delete_model(m)
    models = [x for x in models if x["id"] != model_id]
    save_models(models)
    log.warning("Removed model %s", model_id)
    return {"status": "removed"}


def model_connection_info(m: dict) -> dict:
    """What a client (OpenAI SDK, AnythingLLM, curl) needs to actually talk
    to this model. Always the SAME base_url for every tracked model —
    /v1/chat/completions etc. below dispatch by the request's "model"
    field to whichever Runpod endpoint that model actually lives on, so
    one gateway covers all of them instead of needing a different URL
    per model."""
    deployed = bool((m.get("runpod") or {}).get("endpoint_id"))
    port = os.environ.get("DASHBOARD_PORT", "8787")
    return {
        "base_url": f"http://localhost:{port}/v1",
        "model": m["served_name"],
        "api_key": PROXY_API_KEY or None,
        "ready": deployed,
        "context_window": m.get("recommended_context_window"),
        "max_tokens": m.get("recommended_max_tokens"),
        "note": None if deployed else "Start this model first — it has no live endpoint to route requests to.",
    }


# --------------------------------------------------------- OpenAI gateway --
# One base_url for every model this panel knows about: the "model" field
# in the request picks which Runpod endpoint actually serves it, the same
# way proxy/main.py does for a single model.

def check_gateway_auth(authorization: str | None) -> None:
    if not PROXY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization[len("Bearer "):] != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


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


async def _gateway_dispatch(route: str, body: dict):
    served_name = body.get("model")
    if not served_name:
        raise HTTPException(status_code=400, detail="missing 'model' in request body")
    models = load_models()
    m = next((x for x in models if x["served_name"] == served_name), None)
    if not m:
        raise HTTPException(status_code=404, detail=f"no model named {served_name!r} is tracked by this panel")
    endpoint_id = (m.get("runpod") or {}).get("endpoint_id")
    if not endpoint_id:
        raise HTTPException(status_code=409, detail=f"model {served_name!r} is stopped — Start it first")
    m["last_request_at"] = time.time()  # resets the auto-stop clock, see _check_auto_stop
    save_models(models)
    return await _runpod_job(endpoint_id, route, body)


@app.get("/v1/models")
async def gateway_list_models(authorization: str | None = Header(default=None)):
    check_gateway_auth(authorization)
    now = int(time.time())
    data = []
    for m in load_models():
        if (m.get("runpod") or {}).get("endpoint_id"):
            data.append({"id": m["served_name"], "object": "model", "created": now, "owned_by": "runpod"})
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
