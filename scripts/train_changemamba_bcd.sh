#!/usr/bin/env bash
# ChangeMamba BCD training — defaults live in ./configs/train_changemamba_bcd_vmamba_*.yaml
#
# Two-stage flagship JL1 → joint 512 (one sequential run):
#   ./scripts/train_flagship_bcd_multistage.sh
#
# Usage:
#   ./scripts/train_changemamba_bcd.sh
#   ./scripts/train_changemamba_bcd.sh configs/train_changemamba_bcd_vmamba_small.yaml
#   ./scripts/train_changemamba_bcd.sh configs/train_changemamba_bcd_vmamba_tiny.yaml --batch_size 8
#
# Optional env (launcher only): CUDA_VISIBLE_DEVICES, PYTHON, RUN_FOREGROUND, LOG_DIR, PROJECT_ROOT
# Optional: repo-root .env for SWANLAB_API_KEY, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

DEFAULT_YAML="${PROJECT_ROOT}/configs/train_changemamba_bcd_vmamba_tiny.yaml"

if [[ "${1:-}" == -* ]] || [[ $# -eq 0 ]]; then
  TRAIN_YAML="${DEFAULT_YAML}"
  EXTRA_ARGS=("$@")
else
  TRAIN_YAML="${1}"
  shift
  EXTRA_ARGS=("$@")
fi

if [[ "${TRAIN_YAML}" != /* ]]; then
  TRAIN_YAML="${PROJECT_ROOT}/${TRAIN_YAML}"
fi

if [[ ! -f "${TRAIN_YAML}" ]]; then
  echo "ERROR: Training YAML not found: ${TRAIN_YAML}" >&2
  exit 1
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTHON:=python}"
: "${RUN_FOREGROUND:=0}"
: "${LOG_DIR:=${PROJECT_ROOT}/logs}"
export CUDA_VISIBLE_DEVICES

TRAIN_SCRIPT="${PROJECT_ROOT}/src/ChangeMamba/changedetection/script/train_MambaBCD.py"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=( "${PYTHON}" "${TRAIN_SCRIPT}" --train_config "${TRAIN_YAML}" "${EXTRA_ARGS[@]}" )

echo "Training YAML: ${TRAIN_YAML}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
  "${CMD[@]}"
else
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/train_bcd_$(date +%Y%m%d_%H%M%S).log"
  echo "Background run. Log: ${LOG_FILE}"
  echo "  tail -f ${LOG_FILE}"
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  echo "PID: $!"
fi
