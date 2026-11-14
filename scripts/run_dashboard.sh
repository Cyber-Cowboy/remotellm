#!/usr/bin/env bash
# Local control panel: balance/spend + Start/Stop, run several models
# side by side, and a unified OpenAI-compatible gateway at :8787/v1.
# Binds to 127.0.0.1 only — it holds RUNPOD_API_KEY server-side.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q -r dashboard/requirements.txt
python3 -m uvicorn dashboard.main:app --host 127.0.0.1 --port "${DASHBOARD_PORT:-8787}"
