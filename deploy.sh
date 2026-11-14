#!/usr/bin/env bash
# One-time Runpod bootstrap: creates the shared registry auth + network
# volume + a first template/endpoint. After this, use the dashboard
# (scripts/run_dashboard.sh) to add/start/stop models — it reuses the
# shared volume this creates so models never re-download each other's
# weights.
#
# Usage:
#   set -a; . ./.env; set +a
#   bash deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

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
echo "gateway in front of it — Runpod's job-queue API doesn't speak OpenAI wire"
echo "format directly. Use the dashboard (bash scripts/run_dashboard.sh), which"
echo "runs this gateway itself at http://localhost:8787/v1, or for local testing"
echo "without the dashboard: bash scripts/run_proxy_local.sh"
