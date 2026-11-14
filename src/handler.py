"""
Runpod Serverless handler for a GGUF (llama.cpp) backed OpenAI-compatible model.

Lifecycle per container (worker):
  1. Module import  -> ensure_model_downloaded()  -> ensure_llama_server_running()
     This happens ONCE per cold start. The model is loaded into VRAM here, not
     per-request, so every request handled by an already-running worker is warm.
  2. runpod.serverless.start(...) polls Runpod for jobs and calls `handler` for each.
  3. `handler` forwards the OpenAI-style payload straight to the local llama-server
     and streams tokens back as they arrive (generator -> Runpod's /stream + /run).
  4. When Runpod decides the worker has been idle for `idleTimeout` seconds, it kills
     the container. Nothing here needs to implement shutdown/exit logic.
"""
import json
import time

import httpx
import runpod

from llama_runtime import (
    LLAMA_BASE_URL,
    LLAMA_INTERNAL_API_KEY,
    ensure_llama_server_running,
    ensure_model_downloaded,
    log,
)

_t_cold_start = time.time()
_is_warm = False  # flips to True after the first successfully served request


def _forward_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if LLAMA_INTERNAL_API_KEY:
        h["Authorization"] = f"Bearer {LLAMA_INTERNAL_API_KEY}"
    return h


def handler(job):
    """Runpod generator handler.

    Expected job["input"]:
        {
          "route": "/v1/chat/completions" | "/v1/completions" | "/v1/models",
          "payload": { ...standard OpenAI request body... }
        }

    Yields raw JSON-decodable dicts; Runpod wraps each as {"output": ...} and
    makes them available incrementally via /stream (when return_aggregate_stream
    is set) and in aggregate via /run or /runsync.
    """
    global _is_warm

    job_input = job.get("input", {})
    route = job_input.get("route", "/v1/chat/completions")
    payload = job_input.get("payload", {})
    stream = bool(payload.get("stream", False))

    log.info("request received route=%s stream=%s cold=%s", route, stream, not _is_warm)
    t_start = time.time()

    ensure_llama_server_running()  # cheap no-op if already up; self-heals if it crashed

    url = f"{LLAMA_BASE_URL}{route}"
    headers = _forward_headers()

    try:
        if stream:
            log.info("generation started (stream)")
            first_token_at = None
            n_chunks = 0
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                        else:
                            data = line
                        if data.strip() == "[DONE]":
                            break
                        if first_token_at is None:
                            first_token_at = time.time()
                            log.info("time-to-first-token: %.3fs", first_token_at - t_start)
                        n_chunks += 1
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
            log.info(
                "generation finished chunks=%d total_latency=%.3fs",
                n_chunks, time.time() - t_start,
            )
        else:
            log.info("generation started (non-stream)")
            with httpx.Client(timeout=None) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                yield resp.json()
            log.info("generation finished total_latency=%.3fs", time.time() - t_start)

        _is_warm = True
    except httpx.HTTPStatusError as e:
        log.error("llama-server returned %s: %s", e.response.status_code, e.response.text[:500])
        yield {"error": f"backend error {e.response.status_code}", "detail": e.response.text[:2000]}
    except Exception as e:  # noqa: BLE001 - surface any backend failure to the client
        log.exception("handler failed")
        yield {"error": str(e)}

    log.info("worker idle (waiting for next job or idle-timeout shutdown)")


if __name__ == "__main__":
    log.info("worker booting (cold start)")
    ensure_model_downloaded()
    ensure_llama_server_running()
    log.info("worker ready, total cold-start time %.1fs", time.time() - _t_cold_start)
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
