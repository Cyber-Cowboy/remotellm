"""
OpenAI-compatible gateway in front of a Runpod Serverless endpoint.

Clients talk standard OpenAI wire format to this service:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models
    GET  /health

This service holds RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID; neither is ever
sent to, or visible from, the client. It translates each OpenAI request into
a Runpod job ({"input": {"route": ..., "payload": ...}}), submits it via the
job-based /run, /runsync, /stream Runpod API, and translates the result back
into an OpenAI-shaped (streaming or non-streaming) HTTP response.
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("proxy")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
    raise RuntimeError(
        "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set (see .env.example)"
    )
RUNPOD_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.io/v2")
RUNPOD_BASE = f"{RUNPOD_API_BASE}/{RUNPOD_ENDPOINT_ID}"

PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-uncensored")
ALLOWED_ORIGINS = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]
UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "600"))
RATE_LIMIT_PER_MINUTE = os.environ.get("RATE_LIMIT_PER_MINUTE", "60")
LOG_PROMPTS = os.environ.get("LOG_PROMPTS", "false").lower() == "true"

if not PROXY_API_KEY:
    log.warning("PROXY_API_KEY is not set — the gateway is UNAUTHENTICATED. "
                "Set PROXY_API_KEY before exposing this publicly.")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS)
    yield
    await app.state.http.aclose()


app = FastAPI(title="OpenAI-compatible Runpod gateway", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # never "*" unless explicitly configured
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


def check_auth(authorization: str | None) -> None:
    if not PROXY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization[len("Bearer "):] != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _runpod_headers() -> dict:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


def _extract_output(job_result: dict) -> Any:
    """Normalize Runpod's aggregated generator output into a single dict."""
    output = job_result.get("output")
    if isinstance(output, list):
        return output[-1] if output else {}
    return output or {}


@app.get("/health")
async def health():
    return {"status": "ok", "served_model": SERVED_MODEL_NAME}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "created": now, "owned_by": "self-hosted"}],
    }


async def _run_job(request: Request, route: str, body: dict) -> StreamingResponse | JSONResponse:
    client: httpx.AsyncClient = request.app.state.http
    req_id = str(uuid.uuid4())
    stream = bool(body.get("stream", False))
    runpod_input = {"input": {"route": route, "payload": body}}

    if LOG_PROMPTS:
        log.info("request %s route=%s body=%s", req_id, route, json.dumps(body)[:2000])
    else:
        log.info("request %s route=%s stream=%s model=%s", req_id, route, stream, body.get("model"))

    t0 = time.time()

    if stream:
        async def event_stream():
            submit = await client.post(f"{RUNPOD_BASE}/run", json=runpod_input, headers=_runpod_headers())
            if submit.status_code >= 400:
                log.error("request %s runpod /run failed %s: %s", req_id, submit.status_code, submit.text[:500])
                yield f"data: {json.dumps({'error': 'upstream submit failed'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            job_id = submit.json()["id"]
            first_token_logged = False
            while True:
                r = await client.get(f"{RUNPOD_BASE}/stream/{job_id}", headers=_runpod_headers())
                if r.status_code >= 400:
                    log.error("request %s stream poll failed %s: %s", req_id, r.status_code, r.text[:300])
                    break
                data = r.json()
                for item in data.get("stream", []):
                    out = item.get("output")
                    if out is None:
                        continue
                    if not first_token_logged:
                        log.info("request %s time-to-first-token=%.3fs", req_id, time.time() - t0)
                        first_token_logged = True
                    yield f"data: {json.dumps(out)}\n\n"
                if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                    break
                import asyncio
                await asyncio.sleep(0.15)
            log.info("request %s finished total_latency=%.3fs", req_id, time.time() - t0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: try the synchronous endpoint first (fast path, warm worker).
    r = await client.post(f"{RUNPOD_BASE}/runsync", json=runpod_input, headers=_runpod_headers())
    if r.status_code >= 400:
        log.error("request %s runsync failed %s: %s", req_id, r.status_code, r.text[:500])
        raise HTTPException(status_code=502, detail="upstream error")
    result = r.json()

    # Cold start can exceed runsync's internal wait window; fall back to polling.
    if result.get("status") not in ("COMPLETED",):
        job_id = result.get("id")
        if not job_id:
            raise HTTPException(status_code=502, detail=f"unexpected upstream status: {result.get('status')}")
        while True:
            s = await client.get(f"{RUNPOD_BASE}/status/{job_id}", headers=_runpod_headers())
            result = s.json()
            if result.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            import asyncio
            await asyncio.sleep(0.5)

    if result.get("status") != "COMPLETED":
        log.error("request %s did not complete: %s", req_id, result.get("status"))
        raise HTTPException(status_code=504, detail=f"job {result.get('status', 'unknown')}")

    output = _extract_output(result)
    log.info("request %s finished total_latency=%.3fs", req_id, time.time() - t0)
    if isinstance(output, dict) and "error" in output:
        raise HTTPException(status_code=502, detail=output["error"])
    return JSONResponse(content=output)


@app.post("/v1/chat/completions")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    body = await request.json()
    return await _run_job(request, "/v1/chat/completions", body)


@app.post("/v1/completions")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def completions(request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    body = await request.json()
    return await _run_job(request, "/v1/completions", body)
