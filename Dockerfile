# syntax=docker/dockerfile:1
#
# Runpod Serverless worker: llama.cpp (CUDA) + a thin Python handler.
# Build on a machine/CI runner with real disk (CUDA build artifacts are large) -
# see .github/workflows/docker-build.yml for the recommended path.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# ---------- Stage 1: build llama.cpp with CUDA support ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Pin this to a specific release tag (e.g. "b4433") from
# https://github.com/ggml-org/llama.cpp/releases once you've verified a build
# works for you, so later CI runs can't pull in a breaking change unnoticed.
# "master" is used by default so the very first build always succeeds.
ARG LLAMA_CPP_REF=master
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp.git .

RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release --target llama-server -j"$(nproc)"

# ---------- Stage 2: slim CUDA runtime + Python handler ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/handler.py .

# Defaults; override at endpoint/template creation time via env vars.
ENV MODEL_REPO=HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive \
    MODEL_FILE=GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf \
    MODEL_DIR=/runpod-volume/models/gpt-oss-120b-uncensored \
    SERVED_MODEL_NAME=gpt-oss-120b-uncensored \
    MAX_MODEL_LEN=8192 \
    N_GPU_LAYERS=999 \
    N_CPU_MOE=0 \
    MAX_CONCURRENT=2 \
    LLAMA_SERVER_PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# Not used by Runpod's scheduler (it talks to the handler via its own job
# protocol, not HTTP) but useful for local `docker run` smoke tests.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD curl -fs http://127.0.0.1:8080/health || exit 1

CMD ["python3", "-u", "handler.py"]
