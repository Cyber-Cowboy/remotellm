"""
Shared cold-start logic for the llama.cpp (CUDA) backed OpenAI-compatible
worker: download the GGUF once into persistent storage, then launch
llama-server and block until it reports healthy. Used by src/handler.py
(Runpod Serverless).
"""
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

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

LLAMA_SERVER_HOST = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"  # always loopback for OUR OWN health polling
LLAMA_INTERNAL_API_KEY = os.environ.get("LLAMA_INTERNAL_API_KEY", "")

MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
N_GPU_LAYERS = os.environ.get("N_GPU_LAYERS", "999")
N_CPU_MOE = os.environ.get("N_CPU_MOE", "0")
MAX_CONCURRENT = os.environ.get("MAX_CONCURRENT", "2")

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def _mask(secret: str) -> str:
    if not secret:
        return "<unset>"
    return secret[:4] + "*" * max(0, len(secret) - 8) + secret[-4:]


def ensure_model_downloaded() -> None:
    """Download the GGUF once into persistent storage.

    Safe to call on every cold start: if the file is already there (from a
    previous container run, since the mounted volume persists across
    containers), this is a no-op and costs one stat() call.
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
        "--host", LLAMA_SERVER_HOST,
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
