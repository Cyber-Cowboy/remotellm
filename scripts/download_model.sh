#!/usr/bin/env bash
# One-time model pre-warm.
#
# The worker (src/handler.py) already downloads the model on its own first
# ever cold start if it's missing from the network volume, so this script is
# OPTIONAL — but running it once avoids making your very first real user
# wait through a ~61GB download on top of model load.
#
# Run this from any machine that has the SAME Runpod network volume mounted,
# e.g. a short-lived Runpod Pod created with that network volume attached at
# /workspace (see README "Pre-warm the model" section). Terminate the pod
# once this finishes — you're billed for pod time, not for the volume.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive}"
MODEL_FILE="${MODEL_FILE:-GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf}"
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models/gpt-oss-120b-uncensored}"

echo "Downloading ${MODEL_REPO}/${MODEL_FILE} -> ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

python3 -m pip show huggingface_hub >/dev/null 2>&1 || pip install -q "huggingface_hub[cli]"

HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARG=(--token "${HF_TOKEN}")
fi

huggingface-cli download "${MODEL_REPO}" "${MODEL_FILE}" \
  --local-dir "${MODEL_DIR}" \
  "${HF_TOKEN_ARG[@]}"

echo "Done. Files in ${MODEL_DIR}:"
ls -lh "${MODEL_DIR}"
