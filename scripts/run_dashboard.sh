#!/usr/bin/env bash
# Local control panel: balance/spend + Start/Stop for both providers.
# Binds to 127.0.0.1 only — it holds RUNPOD_API_KEY/MODAL_TOKEN_* server-side.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q -r dashboard/requirements.txt
python3 -m uvicorn dashboard.main:app --host 127.0.0.1 --port "${DASHBOARD_PORT:-8787}"
