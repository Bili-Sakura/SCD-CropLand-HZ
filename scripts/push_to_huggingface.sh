#!/usr/bin/env bash
# Push models/BiliSakura/JL1-ChangeMambaSCD (or LOCAL_DIR) to Hugging Face.
#
# Default: Storage Buckets (hf://buckets/...), not git model repos — see:
#   https://huggingface.co/docs/huggingface_hub/en/guides/buckets
#
# Prerequisites:
#   - huggingface_hub >= 1.5 (see environment.yaml)
#   - HF_TOKEN with write access: https://huggingface.co/settings/tokens
#
# Env:
#   HF_TOKEN            — required (unless you pass --token to the Python CLI)
#   HF_BUCKET_ID        — default BiliSakura/ChangeMambaSCD-JL1-CUP-2024
#   HF_BUCKET_PREFIX    — optional subpath inside the bucket (e.g. v1)
#   HF_PUSH_MODE        — bucket (default) or model
#   HF_MODEL_REPO_ID    — for HF_PUSH_MODE=model
#   LOCAL_DIR           — optional override for folder to upload
#
# Optional: repo-root .env (copy from example.env); or export vars in the shell.
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./scripts/push_to_huggingface.sh
#   ./scripts/push_to_huggingface.sh -- --dry-run
#   ./scripts/push_to_huggingface.sh -- --mode model --repo-id Org/MyModelCardRepo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# Allow: ./push_to_huggingface.sh -- --repo-id …  (strip bash-style end-of-options)
if [[ "${1:-}" == "--" ]]; then
  shift
fi

: "${PYTHON:=python}"
LOCAL_ARGS=()
if [[ -n "${LOCAL_DIR:-}" ]]; then
  LOCAL_ARGS+=(--local-dir "${LOCAL_DIR}")
fi

exec "${PYTHON}" "${SCRIPT_DIR}/push_to_huggingface.py" "${LOCAL_ARGS[@]}" "$@"
