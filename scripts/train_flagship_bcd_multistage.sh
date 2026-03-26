#!/usr/bin/env bash
# Flagship BCD — three-stage pipeline in one shot (see docs/roadmap.md).
#   Checkpoints: models/BiliSakura/ChangeMambaBCD/<stage_subdir>/ (override FLAGSHIP_CKPT_ROOT).
#   Stage 1: JL1 256px, 1k steps (override STAGE1_MAX_ITERS), bs=32 → last_model.pth + periodic ckpt (see train_MambaBCD.py).
#   Stage 2: joint CLCD+FPCD+Hi-CNA at 512px (one manifest, no HRSCD), 5k steps, bs=8, resumes from stage 1.
#   Stage 3: input_quick 512px refinement, 5k steps, bs=8, resumes from stage 2 last_model.pth.
#
# Usage:
#   ./scripts/train_flagship_bcd_multistage.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train_flagship_bcd_multistage.sh
#
# Foreground by default. To detach like train_changemamba_bcd.sh:
#   RUN_BACKGROUND=1 ./scripts/train_flagship_bcd_multistage.sh
# Logs go to LOG_DIR (default: repo logs/).
#
# SwanLab: train_MambaBCD.py calls init_swanlab when SWANLAB_API_KEY is set (see .env).
# Metrics: train/loss every 10 steps; val/* and checkpoint_step_*.pth cadence set by FLAGSHIP_EVAL_INTERVAL / FLAGSHIP_CHECKPOINT_INTERVAL (default 1000 each).
#
# Optional env: PYTHON, PROJECT_ROOT, LOG_DIR, RUN_BACKGROUND, FLAGSHIP_CKPT_ROOT
# Optional: STAGE1_MAX_ITERS (default 1000), STAGE2_MAX_ITERS / STAGE3_MAX_ITERS (default 5000 each)
# Optional: FLAGSHIP_EVAL_INTERVAL / FLAGSHIP_CHECKPOINT_INTERVAL (default 1000)
# Optional: STAGE1_YAML, STAGE2_YAML, STAGE3_YAML
# Optional: STAGE1_LAST_CKPT, STAGE2_LAST_CKPT (override auto stage checkpoints)
# Optional: STOP_AFTER_STAGE1=1 or STOP_AFTER_STAGE2=1
# Stage-2 collections: run reformat separately (tqdm progress), then training:
#   python scripts/reformat_cropland_to_bcd_collections.py --patch_size 512 --output_root datasets/cropland_bcd_collections
# Optional inline prep before Stage 2 (no tqdm in log file unless attached to a TTY):
#   PREP_STAGE2_DATA=1
#   REFORMAT_WORKERS=0          # passed to --workers (<=0 auto, 1 disables multiprocessing)
#   STAGE2_COLLECTIONS_ROOT=...  # default: datasets/cropland_bcd_collections
#   STAGE2_MANIFEST=...          # default: <collections>/train_stage2_joint.txt
#   STAGE2_REQUIRE_SOURCES=1     # require clcd/fpcd/hi_cna and reject hrscd in Stage-2 manifest
#
# Requires model_save_path in each stage YAML under FLAGSHIP_CKPT_ROOT (or set STAGE*_LAST_CKPT).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

: "${LOG_DIR:=${PROJECT_ROOT}/logs}"
: "${RUN_BACKGROUND:=0}"

# Optional: run entire pipeline detached (all stages in one log file)
if [[ "${RUN_BACKGROUND}" == "1" ]] && [[ -z "${_FLAGSHIP_MULTISTAGE_BG_CHILD:-}" ]]; then
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/train_flagship_multistage_$(date +%Y%m%d_%H%M%S).log"
  echo "RUN_BACKGROUND=1 → detached run, log: ${LOG_FILE}"
  echo "  tail -f ${LOG_FILE}"
  nohup env _FLAGSHIP_MULTISTAGE_BG_CHILD=1 bash "${BASH_SOURCE[0]}" >> "${LOG_FILE}" 2>&1 &
  echo "PID: $!"
  exit 0
fi

: "${PYTHON:=python}"
: "${STAGE1_YAML:=${PROJECT_ROOT}/configs/flagship_bcd_stage1_jl1_vmamba_base.yaml}"
: "${STAGE2_YAML:=${PROJECT_ROOT}/configs/flagship_bcd_stage2_joint_vmamba_base.yaml}"
: "${STAGE3_YAML:=${PROJECT_ROOT}/configs/flagship_bcd_stage3_input_quick_vmamba_base.yaml}"
: "${STOP_AFTER_STAGE1:=0}"
: "${STOP_AFTER_STAGE2:=0}"
: "${PREP_STAGE2_DATA:=0}"
: "${STAGE2_REQUIRE_SOURCES:=1}"
: "${REFORMAT_WORKERS:=0}"
# Default: /root/workspace/sakura/SCD-CropLand-HZ/models/BiliSakura/ChangeMambaBCD when PROJECT_ROOT is this repo
: "${FLAGSHIP_CKPT_ROOT:=${PROJECT_ROOT}/models/BiliSakura/ChangeMambaBCD}"
: "${STAGE1_MAX_ITERS:=1000}"
: "${STAGE2_MAX_ITERS:=5000}"
: "${STAGE3_MAX_ITERS:=5000}"
: "${FLAGSHIP_EVAL_INTERVAL:=1000}"
: "${FLAGSHIP_CHECKPOINT_INTERVAL:=1000}"
: "${STAGE1_SAVE_DIR:=${FLAGSHIP_CKPT_ROOT}/FlagshipBaseStage1_JL1}"
: "${STAGE1_LAST_CKPT:=${STAGE1_SAVE_DIR}/last_model.pth}"
: "${STAGE2_SAVE_DIR:=${FLAGSHIP_CKPT_ROOT}/FlagshipBaseStage2_Joint512}"
: "${STAGE2_LAST_CKPT:=${STAGE2_SAVE_DIR}/last_model.pth}"
: "${STAGE2_COLLECTIONS_ROOT:=${PROJECT_ROOT}/datasets/cropland_bcd_collections}"
: "${STAGE2_MANIFEST:=${STAGE2_COLLECTIONS_ROOT}/train_stage2_joint.txt}"
: "${REFORMAT_SCRIPT:=${PROJECT_ROOT}/scripts/reformat_cropland_to_bcd_collections.py}"

TRAIN_SCRIPT="${PROJECT_ROOT}/src/ChangeMamba/changedetection/script/train_MambaBCD.py"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [[ ! -f "${STAGE1_YAML}" ]]; then
  echo "ERROR: Stage 1 YAML not found: ${STAGE1_YAML}" >&2
  exit 1
fi
if [[ "${STOP_AFTER_STAGE1}" != "1" ]]; then
  if [[ ! -f "${STAGE2_YAML}" ]]; then
    echo "ERROR: Stage 2 YAML not found: ${STAGE2_YAML}" >&2
    exit 1
  fi
fi
if [[ "${STOP_AFTER_STAGE1}" != "1" && "${STOP_AFTER_STAGE2}" != "1" ]]; then
  if [[ ! -f "${STAGE3_YAML}" ]]; then
    echo "ERROR: Stage 3 YAML not found: ${STAGE3_YAML}" >&2
    exit 1
  fi
fi

check_stage2_manifest_sources() {
  local manifest_path="$1"
  local has_clcd=0
  local has_fpcd=0
  local has_hi_cna=0
  local has_hrscd=0
  local n_clcd=0
  local n_fpcd=0
  local n_hi_cna=0
  local n_hrscd=0
  local has_any=0
  local line

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    has_any=1
    case "${line}" in
      clcd/*) has_clcd=1; n_clcd=$((n_clcd + 1)) ;;
      fpcd/*) has_fpcd=1; n_fpcd=$((n_fpcd + 1)) ;;
      hi_cna/*) has_hi_cna=1; n_hi_cna=$((n_hi_cna + 1)) ;;
      hrscd/*) has_hrscd=1; n_hrscd=$((n_hrscd + 1)) ;;
    esac
  done < "${manifest_path}"

  if [[ "${has_any}" == "0" ]]; then
    echo "ERROR: Stage 2 manifest is empty: ${manifest_path}" >&2
    return 1
  fi
  if [[ "${has_clcd}" != "1" || "${has_fpcd}" != "1" || "${has_hi_cna}" != "1" ]]; then
    echo "ERROR: Stage 2 manifest must include clcd/fpcd/hi_cna." >&2
    echo "  manifest: ${manifest_path}" >&2
    echo "  seen -> clcd:${has_clcd} fpcd:${has_fpcd} hi_cna:${has_hi_cna}" >&2
    return 1
  fi
  if [[ "${has_hrscd}" == "1" ]]; then
    echo "ERROR: Stage 2 manifest must not include hrscd for this run." >&2
    echo "  manifest: ${manifest_path}" >&2
    echo "  hrscd entries: ${n_hrscd}" >&2
    return 1
  fi
  echo "Stage 2 manifest check OK: includes clcd/fpcd/hi_cna and excludes hrscd"
  echo "  counts -> clcd:${n_clcd} fpcd:${n_fpcd} hi_cna:${n_hi_cna} hrscd:${n_hrscd}"
}

echo "========== Stage 1 / JL1 (256px, ${STAGE1_MAX_ITERS} iters) =========="
echo "Config: ${STAGE1_YAML}"
"${PYTHON}" "${TRAIN_SCRIPT}" --train_config "${STAGE1_YAML}" \
  --max_iters "${STAGE1_MAX_ITERS}" \
  --eval_interval "${FLAGSHIP_EVAL_INTERVAL}" \
  --checkpoint_interval "${FLAGSHIP_CHECKPOINT_INTERVAL}"

if [[ ! -f "${STAGE1_LAST_CKPT}" ]]; then
  echo "ERROR: Stage 1 did not write ${STAGE1_LAST_CKPT}" >&2
  exit 1
fi

if [[ "${STOP_AFTER_STAGE1}" == "1" ]]; then
  echo ""
  echo "Stopped after Stage 1 by STOP_AFTER_STAGE1=1"
  echo "Stage 1 checkpoint: ${STAGE1_LAST_CKPT}"
  echo ""
  echo "Manual next step (Stage 2):"
  echo "  ${PYTHON} ${REFORMAT_SCRIPT} --patch_size 512 --output_root ${STAGE2_COLLECTIONS_ROOT}"
  echo "  ${PYTHON} ${TRAIN_SCRIPT} --train_config ${STAGE2_YAML} --resume ${STAGE1_LAST_CKPT} \\"
  echo "    --max_iters ${STAGE2_MAX_ITERS} --eval_interval ${FLAGSHIP_EVAL_INTERVAL} \\"
  echo "    --checkpoint_interval ${FLAGSHIP_CHECKPOINT_INTERVAL}"
  exit 0
fi

if [[ "${PREP_STAGE2_DATA}" == "1" ]]; then
  if [[ ! -f "${REFORMAT_SCRIPT}" ]]; then
    echo "ERROR: Reformat script not found: ${REFORMAT_SCRIPT}" >&2
    exit 1
  fi
  echo ""
  echo "========== Stage 2 data prep / reformat (patch_size=512) =========="
  echo "Script: ${REFORMAT_SCRIPT}"
  echo "Output: ${STAGE2_COLLECTIONS_ROOT}"
  REFORMAT_EXTRA=()
  if [[ ! -t 1 ]]; then
    REFORMAT_EXTRA+=(--no-progress)
  fi
  "${PYTHON}" "${REFORMAT_SCRIPT}" --patch_size 512 --workers "${REFORMAT_WORKERS}" --output_root "${STAGE2_COLLECTIONS_ROOT}" "${REFORMAT_EXTRA[@]}"
fi

if [[ ! -f "${STAGE2_MANIFEST}" ]]; then
  echo "ERROR: Stage 2 manifest not found: ${STAGE2_MANIFEST}" >&2
  echo "Run reformat first (shows tqdm in an interactive terminal):" >&2
  echo "  ${PYTHON} ${REFORMAT_SCRIPT} --patch_size 512 --workers ${REFORMAT_WORKERS} --output_root ${STAGE2_COLLECTIONS_ROOT}" >&2
  echo "Or set PREP_STAGE2_DATA=1 to run it from this script before Stage 2." >&2
  exit 1
fi
if [[ "${STAGE2_REQUIRE_SOURCES}" == "1" ]]; then
  check_stage2_manifest_sources "${STAGE2_MANIFEST}"
fi

echo ""
echo "========== Stage 2 / joint 512 (${STAGE2_MAX_ITERS} iters, resume stage 1) =========="
echo "Config: ${STAGE2_YAML}"
echo "Resume: ${STAGE1_LAST_CKPT}"
"${PYTHON}" "${TRAIN_SCRIPT}" --train_config "${STAGE2_YAML}" --resume "${STAGE1_LAST_CKPT}" \
  --max_iters "${STAGE2_MAX_ITERS}" \
  --eval_interval "${FLAGSHIP_EVAL_INTERVAL}" \
  --checkpoint_interval "${FLAGSHIP_CHECKPOINT_INTERVAL}"

if [[ ! -f "${STAGE2_LAST_CKPT}" ]]; then
  echo "ERROR: Stage 2 did not write ${STAGE2_LAST_CKPT}" >&2
  exit 1
fi

if [[ "${STOP_AFTER_STAGE2}" == "1" ]]; then
  echo ""
  echo "Stopped after Stage 2 by STOP_AFTER_STAGE2=1"
  echo "Stage 2 checkpoint: ${STAGE2_LAST_CKPT}"
  echo ""
  echo "Manual next step (Stage 3):"
  echo "  ${PYTHON} ${TRAIN_SCRIPT} --train_config ${STAGE3_YAML} --resume ${STAGE2_LAST_CKPT} \\"
  echo "    --max_iters ${STAGE3_MAX_ITERS} --eval_interval ${FLAGSHIP_EVAL_INTERVAL} \\"
  echo "    --checkpoint_interval ${FLAGSHIP_CHECKPOINT_INTERVAL}"
  exit 0
fi

echo ""
echo "========== Stage 3 / input_quick 512 refinement (${STAGE3_MAX_ITERS} iters, resume stage 2) =========="
echo "Config: ${STAGE3_YAML}"
echo "Resume: ${STAGE2_LAST_CKPT}"
"${PYTHON}" "${TRAIN_SCRIPT}" --train_config "${STAGE3_YAML}" --resume "${STAGE2_LAST_CKPT}" \
  --max_iters "${STAGE3_MAX_ITERS}" \
  --eval_interval "${FLAGSHIP_EVAL_INTERVAL}" \
  --checkpoint_interval "${FLAGSHIP_CHECKPOINT_INTERVAL}"

echo ""
echo "Done. Final checkpoints: see model_save_path in ${STAGE3_YAML}"
