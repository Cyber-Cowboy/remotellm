#!/usr/bin/env bash
# Idempotent Modal bootstrap + deploy: auth, registry/worker secrets, then
# `modal deploy modal_app.py`. Source your .env first:
#
#   set -a; . ./.env; set +a
#   bash deploy/modal_deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODAL_TOKEN_ID:?set MODAL_TOKEN_ID in .env}"
: "${MODAL_TOKEN_SECRET:?set MODAL_TOKEN_SECRET in .env}"
: "${DOCKERHUB_USERNAME:?set DOCKERHUB_USERNAME in .env}"
: "${DOCKERHUB_TOKEN:?set DOCKERHUB_TOKEN in .env}"

python3 -m modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

python3 -m modal secret create dockerhub-registry-secret \
  REGISTRY_USERNAME="$DOCKERHUB_USERNAME" \
  REGISTRY_PASSWORD="$DOCKERHUB_TOKEN" \
  --force

python3 -m modal secret create gptoss-worker-secrets \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-120b-uncensored}" \
  MODEL_REPO="${MODEL_REPO:-HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive}" \
  MODEL_FILE="${MODEL_FILE:-GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf}" \
  MODEL_DIR="/cache/models/gpt-oss-120b-uncensored" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
  N_GPU_LAYERS="${N_GPU_LAYERS:-999}" \
  N_CPU_MOE="${N_CPU_MOE:-0}" \
  MAX_CONCURRENT="${MAX_CONCURRENT:-2}" \
  LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8080}" \
  LLAMA_INTERNAL_API_KEY="${LLAMA_INTERNAL_API_KEY:-}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  PROXY_API_KEY="${PROXY_API_KEY:-}" \
  ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-}" \
  RATE_LIMIT_PER_MINUTE="${RATE_LIMIT_PER_MINUTE:-60}" \
  UPSTREAM_TIMEOUT_SECONDS="${UPSTREAM_TIMEOUT_SECONDS:-600}" \
  LOG_PROMPTS="${LOG_PROMPTS:-false}" \
  --force

python3 -m modal deploy modal_app.py 2>&1 | tee /tmp/modal_deploy_output.$$.log
url=$(grep -oE 'https://[A-Za-z0-9.-]*modal\.run[A-Za-z0-9/_-]*' /tmp/modal_deploy_output.$$.log | head -1 || true)
rm -f /tmp/modal_deploy_output.$$.log
if [ -n "$url" ]; then
  python3 -c "import json; json.dump({'web_url': '$url'}, open('deploy/modal_state.json', 'w'), indent=2)"
  echo
  echo "Base URL for clients (OpenAI SDK base_url, append /v1): $url/v1"
else
  echo
  echo "Could not auto-detect the web URL from deploy output — check it with:"
  echo "  python3 -m modal app list"
fi
