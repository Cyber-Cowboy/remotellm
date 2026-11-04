#!/usr/bin/env bash
# End-to-end scale-to-zero verification, per the 11-step checklist:
#   workers=0 -> request -> worker starts -> warm request -> idle ->
#   workers=0 again -> request -> worker starts again.
#
# Talks directly to the Runpod job API (bypasses the proxy) so it isolates
# Runpod/worker behaviour from the gateway.
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export RUNPOD_ENDPOINT_ID=...
#   ./scripts/test_endpoint.sh
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID}"
BASE="https://api.runpod.io/v2/${RUNPOD_ENDPOINT_ID}"
AUTH=(-H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json")
IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"

PAYLOAD='{"input":{"route":"/v1/chat/completions","payload":{"model":"gpt-oss-120b-uncensored","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":32}}}'

workers() {
  curl -s "${AUTH[@]}" "${BASE}/health" | tee /dev/stderr
}

echo "== 1. workers before any traffic =="
workers; echo

echo "== 2/3. sending first (cold) request, timing it =="
t0=$(date +%s.%N)
run_id=$(curl -s "${AUTH[@]}" -d "${PAYLOAD}" "${BASE}/run" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
echo "job id: ${run_id}"
while true; do
  status_json=$(curl -s "${AUTH[@]}" "${BASE}/status/${run_id}")
  status=$(echo "${status_json}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')
  [[ "${status}" == "COMPLETED" || "${status}" == "FAILED" ]] && break
  sleep 1
done
t1=$(date +%s.%N)
echo "cold request status=${status} latency=$(echo "$t1 - $t0" | bc)s"
echo "${status_json}" | python3 -m json.tool | head -40

echo "== 4. worker should now show >=1 =="
workers; echo

echo "== 5/6. sending second (warm) request, timing it =="
t0=$(date +%s.%N)
run_id2=$(curl -s "${AUTH[@]}" -d "${PAYLOAD}" "${BASE}/run" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
while true; do
  status=$(curl -s "${AUTH[@]}" "${BASE}/status/${run_id2}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')
  [[ "${status}" == "COMPLETED" || "${status}" == "FAILED" ]] && break
  sleep 0.5
done
t1=$(date +%s.%N)
echo "warm request status=${status} latency=$(echo "$t1 - $t0" | bc)s  (should be much lower than cold)"

echo "== 7/8. no more requests — waiting idleTimeout(+buffer) = $((IDLE_TIMEOUT + 30))s for scale-down =="
sleep $((IDLE_TIMEOUT + 30))

echo "== 9. worker should be back to 0 =="
workers; echo

echo "== 10/11. sending a third request — worker must auto-start again =="
t0=$(date +%s.%N)
run_id3=$(curl -s "${AUTH[@]}" -d "${PAYLOAD}" "${BASE}/run" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
while true; do
  status=$(curl -s "${AUTH[@]}" "${BASE}/status/${run_id3}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')
  [[ "${status}" == "COMPLETED" || "${status}" == "FAILED" ]] && break
  sleep 1
done
t1=$(date +%s.%N)
echo "third request status=${status} latency=$(echo "$t1 - $t0" | bc)s (auto scale-up confirmed if COMPLETED)"

echo
echo "Scale-to-zero cycle test finished. Review the 'workers' snapshots above:"
echo "  step 1 should show 0, step 4 should show >=1, step 9 should show 0 again."
