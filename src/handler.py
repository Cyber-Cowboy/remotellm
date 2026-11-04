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
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import runpod

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")

MODEL_REPO = os.environ.get("MODEL_REPO", "HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive")
MODEL_FILE = os.environ.get("MODEL_FILE", "GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/models/gpt-oss-120b-uncensored"))
MODEL_PATH = MODEL_DIR / MODEL_FILE
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-uncensored")

LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
LLAMA_INTERNAL_API_KEY = os.environ.get("LLAMA_INTERNAL_API_KEY", "")

MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
N_GPU_LAYERS = os.environ.get("N_GPU_LAYERS", "999")
N_CPU_MOE = os.environ.get("N_CPU_MOE", "0")
MAX_CONCURRENT = os.environ.get("MAX_CONCURRENT", "2")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

_t_cold_start = time.time()
_is_warm = False  # flips to True after the first successfully served request


def _mask(secret: str) -> str:
    if not secret:
        return "<unset>"
    return secret[:4] + "*" * max(0, len(secret) - 8) + secret[-4:]


def ensure_model_downloaded() -> None:
    """Download the GGUF once into persistent network-volume storage.

    Safe to call on every cold start: if the file is already there (from a
    previous worker run, since /runpod-volume persists across workers), this
    is a no-op and costs one stat() call.
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        log.info("model already present at %s (%.1f GB) — skipping download",
                  MODEL_PATH, MODEL_PATH.stat().st_size / 1e9)
        return

    log.info("model not found at %s — downloading %s/%s once into persistent storage "
              "(this only happens on the very first cold start ever)",
              MODEL_PATH, MODEL_REPO, MODEL_FILE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
        log.info("using HF_TOKEN=%s for gated/private download", _mask(HF_TOKEN))

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
        **kwargs,
    )
    downloaded_path = Path(downloaded_path)
    if downloaded_path != MODEL_PATH:
        downloaded_path.rename(MODEL_PATH)

    log.info("model download finished in %.1fs (%.1f GB)",
              time.time() - t0, MODEL_PATH.stat().st_size / 1e9)


_llama_process: subprocess.Popen | None = None


def ensure_llama_server_running() -> None:
    """Start llama-server once per container and block until it is healthy."""
    global _llama_process

    if _llama_process is not None and _llama_process.poll() is None:
        return  # already running

    cmd = [
        "llama-server",
        "--model", str(MODEL_PATH),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_SERVER_PORT),
        "--ctx-size", str(MAX_MODEL_LEN),
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--n-cpu-moe", str(N_CPU_MOE),
        "--parallel", str(MAX_CONCURRENT),
        "--flash-attn", "on",
        "--jinja",
        "--alias", SERVED_MODEL_NAME,
    ]
    if LLAMA_INTERNAL_API_KEY:
        cmd += ["--api-key", LLAMA_INTERNAL_API_KEY]

    log.info("starting llama-server: %s", " ".join(cmd))
    t0 = time.time()
    _llama_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream llama-server's own logs into our structured logs.
    import threading

    def _pump():
        for line in _llama_process.stdout:  # type: ignore[union-attr]
            log.info("[llama-server] %s", line.rstrip())

    threading.Thread(target=_pump, daemon=True).start()

    headers = {}
    if LLAMA_INTERNAL_API_KEY:
        headers["Authorization"] = f"Bearer {LLAMA_INTERNAL_API_KEY}"

    deadline = time.time() + 1800  # generous: cold load of a 61GB file can be slow
    with httpx.Client(timeout=5) as client:
        while time.time() < deadline:
            if _llama_process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {_llama_process.returncode}"
                )
            try:
                r = client.get(f"{LLAMA_BASE_URL}/health", headers=headers)
                if r.status_code == 200:
                    log.info("model ready — llama-server healthy after %.1fs", time.time() - t0)
                    return
            except httpx.TransportError:
                pass
            time.sleep(2)

    raise RuntimeError("llama-server did not become healthy within 30 minutes")


# ---- cold start: do the heavy lifting once when the process starts, not
# per-request. Guarded by __main__ so this module can be imported (e.g. by
# tests) without immediately trying to download 61GB and launch llama-server.


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
