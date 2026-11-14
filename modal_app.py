"""
Modal deployment: OpenAI-compatible endpoint -> Modal Serverless (GPU,
scale-to-zero) -> llama-server (CUDA) -> gpt-oss-120b GGUF.

One container does double duty:
  - @modal.enter() runs once per cold start: download the GGUF into a
    persistent Modal Volume (skipped on every later cold start) and launch
    llama-server bound to 127.0.0.1:8080.
  - @modal.asgi_app() exposes a small FastAPI gateway (src/modal_gateway.py)
    that Modal routes all HTTPS traffic to: it checks the bearer token,
    rate-limits, restricts CORS, and reverse-proxies straight to the local
    llama-server, streaming included. This collapses what was two separate
    services on Runpod (worker + proxy) into one container here.

Deploy with:
    set -a; . ./.env; set +a
    modal deploy modal_app.py

Prereqs (see README.md "Modal deployment"):
    modal token set --token-id ... --token-secret ...
    modal secret create dockerhub-registry-secret REGISTRY_USERNAME=... REGISTRY_PASSWORD=...
    modal secret create gptoss-worker-secrets SERVED_MODEL_NAME=... MODEL_REPO=... ...

Running more than one model side by side (see dashboard/main.py, which
drives this automatically): set MODAL_APP_SUFFIX to something like
"-mymodel" and MODAL_WORKER_SECRET_NAME to a secret created for that
model, then `modal deploy` again — this produces a second, independent
Modal App (e.g. "gptoss-120b-mymodel") rather than overwriting the first.
All models still share the registry secret and the model-cache volume
(each under its own /cache/models/<served_name> subdirectory, set via
that model's secret's MODEL_DIR), so nothing downloaded is ever
duplicated or lost when you add or remove a model.
"""
import os
import sys

import modal

DOCKERHUB_USERNAME = os.environ.get("DOCKERHUB_USERNAME", "")
WORKER_IMAGE_REPO = os.environ.get("MODAL_WORKER_IMAGE", f"docker.io/{DOCKERHUB_USERNAME}/gptoss-120b-worker:latest")
GPU_TYPES = [g.strip() for g in os.environ.get("MODAL_GPU_TYPES", "A100-80GB,H100").split(",") if g.strip()]
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "1"))
SCALEDOWN_WINDOW = int(os.environ.get("MODAL_SCALEDOWN_WINDOW_SECONDS", "60"))
STARTUP_TIMEOUT = int(os.environ.get("MODAL_STARTUP_TIMEOUT_SECONDS", "1800"))  # covers first-ever model download
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "gptoss-120b-model-cache")

APP_SUFFIX = os.environ.get("MODAL_APP_SUFFIX", "")  # e.g. "-mymodel" for a second/third/... model
WORKER_SECRET_NAME = os.environ.get("MODAL_WORKER_SECRET_NAME", "gptoss-worker-secrets")

app = modal.App(f"gptoss-120b{APP_SUFFIX}")

registry_secret = modal.Secret.from_name("dockerhub-registry-secret")
worker_secret = modal.Secret.from_name(WORKER_SECRET_NAME)

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(WORKER_IMAGE_REPO, secret=registry_secret, add_python="3.11")
    .pip_install(
        "httpx>=0.27.0",
        "huggingface_hub>=0.25.0",
        "fastapi>=0.115.0",
        "slowapi>=0.1.9",
    )
)


@app.cls(
    image=image,
    gpu=GPU_TYPES,
    volumes={"/cache": model_volume},
    secrets=[worker_secret],
    min_containers=0,            # Active Workers = 0: scale-to-zero, no idle GPU billing
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=600,                  # per-request ceiling (long generations)
    startup_timeout=STARTUP_TIMEOUT,
)
@modal.concurrent(max_inputs=int(os.environ.get("MAX_CONCURRENT", "2")))
class Worker:
    @modal.enter()
    def setup(self):
        sys.path.insert(0, "/app")
        from llama_runtime import (
            LLAMA_BASE_URL,
            LLAMA_INTERNAL_API_KEY,
            ensure_llama_server_running,
            ensure_model_downloaded,
            log,
        )

        self._llama_base_url = LLAMA_BASE_URL
        self._llama_internal_api_key = LLAMA_INTERNAL_API_KEY

        log.info("container booting (cold start)")
        ensure_model_downloaded()
        model_volume.commit()  # make the downloaded file visible to future containers
        ensure_llama_server_running()
        log.info("container ready")

    @modal.asgi_app()
    def web(self):
        from modal_gateway import build_app

        return build_app(self._llama_base_url, self._llama_internal_api_key)
