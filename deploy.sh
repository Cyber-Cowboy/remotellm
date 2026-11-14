#!/usr/bin/env bash
# Single entry point for deploying to whichever provider is active.
#
# Usage:
#   set -a; . ./.env; set +a
#   bash deploy.sh            # uses PROVIDER from .env
#   PROVIDER=modal bash deploy.sh   # one-off override
#
# See README.md "Switching providers". Both providers serve the exact same
# worker image/model — this script only decides which cloud orchestrates it.
set -euo pipefail
cd "$(dirname "$0")"

PROVIDER="${PROVIDER:?set PROVIDER=runpod|modal in .env, or pass PROVIDER=... on the command line}"

case "$PROVIDER" in
  runpod)
    : "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
    : "${DOCKERHUB_USERNAME:?set DOCKERHUB_USERNAME in .env}"
    IFS=',' read -ra GPU_IDS <<< "${RUNPOD_GPU_TYPE_IDS:-NVIDIA A100 80GB PCIe,NVIDIA H100 PCIe}"
    python3 -I deploy/create_infra.py \
      --image "docker.io/${DOCKERHUB_USERNAME}/gptoss-120b-worker:latest" \
      --name gptoss-120b \
      --data-center "${RUNPOD_DATA_CENTER:-US-CA-2}" \
      --volume-size-gb "${RUNPOD_NETWORK_VOLUME_SIZE_GB:-100}" \
      --gpu-type-ids "${GPU_IDS[@]}"
    echo
    echo "Runpod worker is up. This path needs a separately-running OpenAI-compatible"
    echo "gateway in front of it (Modal bundles this; Runpod's job-queue API doesn't"
    echo "speak OpenAI wire format directly). For local testing:"
    echo "  bash scripts/run_proxy_local.sh"
    echo "then point clients/benchmark.py at PROXY_BASE_URL=http://localhost:8000/v1"
    ;;
  modal)
    : "${MODAL_TOKEN_ID:?set MODAL_TOKEN_ID in .env}"
    : "${DOCKERHUB_USERNAME:?set DOCKERHUB_USERNAME in .env}"
    bash deploy/modal_deploy.sh
    ;;
  *)
    echo "Unknown PROVIDER='$PROVIDER' (expected 'runpod' or 'modal')" >&2
    exit 1
    ;;
esac
