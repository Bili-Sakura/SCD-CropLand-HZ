#!/usr/bin/env bash
# Flagship BCD — two-stage pipeline in one shot (see docs/roadmap.md).
#   Stage 1: JL1 256px, 40k steps, bs=8 → saves last_model.pth + best-kappa checkpoints.
#   Stage 2: joint CLCD 512px (manifest), 50k steps, bs=2, resumes from stage 1 last_model.pth.
#
# Usage:
#   ./scripts/train_flagship_bcd_multistage.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train_flagship_bcd_multistage.sh
#
# Optional env: PYTHON, PROJECT_ROOT
# Optional: STAGE1_YAML, STAGE2_YAML, STAGE1_LAST_CKPT (override checkpoint path after stage 1)
#
# Requires model_save_path in stage-1 YAML to match STAGE1_SAVE_DIR below (or set STAGE1_LAST_CKPT).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

: "${PYTHON:=python}"
: "${STAGE1_YAML:=${PROJECT_ROOT}/configs/flagship_bcd_stage1_jl1_vmamba_base.yaml}"
: "${STAGE2_YAML:=${PROJECT_ROOT}/configs/flagship_bcd_stage2_joint_vmamba_base.yaml}"
# Default matches base stage-1 YAML model_save_path
: "${STAGE1_SAVE_DIR:=${PROJECT_ROOT}/models/BiliSakura/ChangeMambaBCD/FlagshipBaseStage1_JL1}"
: "${STAGE1_LAST_CKPT:=${STAGE1_SAVE_DIR}/last_model.pth}"

TRAIN_SCRIPT="${PROJECT_ROOT}/src/ChangeMamba/changedetection/script/train_MambaBCD.py"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [[ ! -f "${STAGE1_YAML}" ]]; then
  echo "ERROR: Stage 1 YAML not found: ${STAGE1_YAML}" >&2
  exit 1
fi
if [[ ! -f "${STAGE2_YAML}" ]]; then
  echo "ERROR: Stage 2 YAML not found: ${STAGE2_YAML}" >&2
  exit 1
fi

echo "========== Stage 1 / JL1 (256px, 40k iters) =========="
echo "Config: ${STAGE1_YAML}"
"${PYTHON}" "${TRAIN_SCRIPT}" --train_config "${STAGE1_YAML}"

if [[ ! -f "${STAGE1_LAST_CKPT}" ]]; then
  echo "ERROR: Stage 1 did not write ${STAGE1_LAST_CKPT}" >&2
  exit 1
fi

echo ""
echo "========== Stage 2 / joint 512 (50k iters, resume stage 1) =========="
echo "Config: ${STAGE2_YAML}"
echo "Resume: ${STAGE1_LAST_CKPT}"
"${PYTHON}" "${TRAIN_SCRIPT}" --train_config "${STAGE2_YAML}" --resume "${STAGE1_LAST_CKPT}"

echo ""
echo "Done. Stage 2 checkpoints: see model_save_path in ${STAGE2_YAML}"
