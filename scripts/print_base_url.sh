#!/usr/bin/env bash
# Prints the OpenAI-compatible base_url for whichever PROVIDER is active,
# so testing/benchmarking never has to care which cloud is behind it.
set -euo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:?set PROVIDER=runpod|modal in .env}"

case "$PROVIDER" in
  modal)
    if [ -f deploy/modal_state.json ]; then
      python3 -c "import json; print(json.load(open('deploy/modal_state.json'))['web_url'] + '/v1')"
    else
      echo "No deploy/modal_state.json yet — run: bash deploy.sh (PROVIDER=modal)" >&2
      exit 1
    fi
    ;;
  runpod)
    echo "http://localhost:${PROXY_PORT:-8000}/v1"
    echo "(run 'bash scripts/run_proxy_local.sh' first, or deploy proxy/ elsewhere" >&2
    echo " and use that host instead — see README.md 'Runpod deployment')" >&2
    ;;
  *)
    echo "Unknown PROVIDER='$PROVIDER'" >&2
    exit 1
    ;;
esac
