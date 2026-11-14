#!/usr/bin/env bash
# Runs the Runpod OpenAI-compatible gateway (proxy/main.py) locally with
# plain uvicorn — no Docker needed. Use this to test PROVIDER=runpod with
# the OpenAI SDK / scripts/benchmark.py the same way a direct Modal URL
# works for PROVIDER=modal: point them at http://localhost:8000/v1.
#
# For a real deployment, run proxy/ on a real host behind TLS instead
# (see README.md "Runpod deployment" — Deploy the proxy).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
: "${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID in .env (see deploy/infra_state.json after deploy.sh)}"

pip install -q -r proxy/requirements.txt
cd proxy
python3 -m uvicorn main:app --host 0.0.0.0 --port "${PROXY_PORT:-8000}"
