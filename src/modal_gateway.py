"""
OpenAI-compatible FastAPI gateway that sits in front of the local
llama-server sidecar inside the same Modal container.

Unlike the Runpod proxy (proxy/main.py), which has to translate OpenAI
requests into Runpod's job-queue API because the worker and the gateway are
two separate services, this gateway and llama-server live in the same
container on localhost — so this is a thin, direct reverse proxy: check
auth/rate-limit/CORS, forward the request body as-is to llama-server (which
already speaks OpenAI wire format), and stream the response straight back.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

log = logging.getLogger("gateway")

PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-uncensored")
ALLOWED_ORIGINS = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]
RATE_LIMIT_PER_MINUTE = os.environ.get("RATE_LIMIT_PER_MINUTE", "60")
UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "600"))
LOG_PROMPTS = os.environ.get("LOG_PROMPTS", "false").lower() == "true"

if not PROXY_API_KEY:
    log.warning("PROXY_API_KEY is not set — the gateway is UNAUTHENTICATED. "
                "Set PROXY_API_KEY before exposing this publicly.")

limiter = Limiter(key_func=get_remote_address)


def build_app(llama_base_url: str, llama_internal_api_key: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS)
        yield
        await app.state.http.aclose()

    app = FastAPI(title="OpenAI-compatible gateway (Modal)", lifespan=lifespan)
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

    def _upstream_headers() -> dict:
        h = {"Content-Type": "application/json"}
        if llama_internal_api_key:
            h["Authorization"] = f"Bearer {llama_internal_api_key}"
        return h

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

    async def _forward(request: Request, route: str) -> StreamingResponse | JSONResponse:
        client: httpx.AsyncClient = request.app.state.http
        body = await request.json()
        stream = bool(body.get("stream", False))
        url = f"{llama_base_url}{route}"

        if LOG_PROMPTS:
            log.info("request route=%s body=%s", route, json.dumps(body)[:2000])
        else:
            log.info("request route=%s stream=%s model=%s", route, stream, body.get("model"))

        t0 = time.time()

        if stream:
            async def event_stream():
                first_token_at = None
                async with client.stream("POST", url, json=body, headers=_upstream_headers()) as resp:
                    if resp.status_code >= 400:
                        detail = await resp.aread()
                        log.error("llama-server returned %s: %s", resp.status_code, detail[:500])
                        yield f"data: {json.dumps({'error': 'backend error'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            if line[len('data: '):].strip() == "[DONE]":
                                break
                            if first_token_at is None:
                                first_token_at = time.time()
                                log.info("time-to-first-token=%.3fs", first_token_at - t0)
                            yield f"{line}\n\n"
                log.info("generation finished total_latency=%.3fs", time.time() - t0)
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        resp = await client.post(url, json=body, headers=_upstream_headers())
        if resp.status_code >= 400:
            log.error("llama-server returned %s: %s", resp.status_code, resp.text[:500])
            raise HTTPException(status_code=502, detail="upstream error")
        log.info("generation finished total_latency=%.3fs", time.time() - t0)
        return JSONResponse(content=resp.json())

    @app.post("/v1/chat/completions")
    @limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
        check_auth(authorization)
        return await _forward(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    @limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
    async def completions(request: Request, authorization: str | None = Header(default=None)):
        check_auth(authorization)
        return await _forward(request, "/v1/completions")

    return app
