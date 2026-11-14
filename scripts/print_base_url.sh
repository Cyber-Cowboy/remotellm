#!/usr/bin/env bash
# Prints the OpenAI-compatible base_url. The dashboard (scripts/run_dashboard.sh)
# runs its own gateway covering every model you've added through it — that's
# the one to use for anything but a quick one-off test of a single endpoint
# without the dashboard running.
echo "http://localhost:${DASHBOARD_PORT:-8787}/v1"
echo "(needs 'bash scripts/run_dashboard.sh' running; for a single endpoint" >&2
echo " without the dashboard, use scripts/run_proxy_local.sh -> :8000/v1 instead)" >&2
