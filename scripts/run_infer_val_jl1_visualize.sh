#!/usr/bin/env bash
# JL1 val inference + visualization (requires GPU + VMamba CUDA ops; use conda env mambascd).
# Default --resume is best_model.pth (weights only). You can also pass {step}_model.pth
# (contains optimizer, scheduler, …); the script loads only the "model" tensor dict.
#
# Usage:
#   conda activate mambascd
#   ./scripts/run_infer_val_jl1_visualize.sh              # default: --max-samples 8
#   ./scripts/run_infer_val_jl1_visualize.sh --random-one # one random val tile (ignores the 8-cap pool)
#   ./scripts/run_infer_val_jl1_visualize.sh --max-samples 0   # 0 = full val set
#   ./scripts/run_infer_val_jl1_visualize.sh --out-dir results/my_vis
#
# Optional env: PYTHON, CUDA_VISIBLE_DEVICES

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
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Prepend default cap; pass e.g. --max-samples 0 to run the full val set (last flag wins).
exec "${PYTHON}" "${SCRIPT_DIR}/infer_val_jl1_visualize.py" --max-samples 8 "$@"
