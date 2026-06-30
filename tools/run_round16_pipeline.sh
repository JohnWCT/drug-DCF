#!/usr/bin/env bash
# Round 16 pipeline (stage list via ROUND16_PIPELINE_STAGES).
# Default: architecture / feature-family first (16F, 16E). Downstream 16A–16C deferred.
set -euo pipefail

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
# Default: architecture / feature-family first (16F, 16E, 16D). Downstream 16A–16C deferred.
export ROUND16_PIPELINE_STAGES="${ROUND16_PIPELINE_STAGES:-16f,16e,16d}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
LOG_DIR="logs"
CURRENT_STAGE=""

mkdir -p "${LOG_DIR}"

declare -A STAGE_SCRIPT=(
  [16f]="tools/run_round16_delta_replacement_stage16f.sh"
  [16e]="tools/run_round16_own_proto_context_stage16e.sh"
  [16d]="tools/run_round16_pretrain_vicreg_stage16d.sh"
  [16a]="tools/run_round16_bruteforce_stage16a.sh"
  [16b]="tools/run_round16_confirmation_stage16b.sh"
  [16c]="tools/run_round16_feature_variants_stage16c.sh"
)

notify() {
  python3 tools/round16_telegram_notify.py "$@" || true
}

on_exit() {
  local code=$?
  if [[ ${code} -ne 0 && -n "${CURRENT_STAGE}" ]]; then
    notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code ${code}"
    notify --event pipeline-fail --reason "stage ${CURRENT_STAGE} failed (exit ${code})"
  fi
}
trap on_exit EXIT

if [[ "${ROUND16_RESET_MANIFESTS:-0}" == "1" ]]; then
  echo "Resetting Round 16 finetune manifests to pending..."
  python3 tools/round16_telegram_notify.py --event reset-manifests
fi

echo "========== ROUND16 PIPELINE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "ROUND16_PIPELINE_STAGES=${ROUND16_PIPELINE_STAGES}"
echo "FINETUNE_PARALLEL=${FINETUNE_PARALLEL}"
echo "FINETUNE_BATCH_SIZE=${FINETUNE_BATCH_SIZE}"
echo "FINETUNE_MINI_BATCH_SIZE=${FINETUNE_MINI_BATCH_SIZE}"
echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"
if [[ -f config/round16_defer_downstream.flag ]]; then
  echo "Downstream 16A–16C deferred (config/round16_defer_downstream.flag present)"
fi

notify --event pipeline-start --stages "${ROUND16_PIPELINE_STAGES}"

run_stage() {
  local stage="$1"
  local script="$2"
  local log="${LOG_DIR}/round16_${stage}.log"
  CURRENT_STAGE="${stage}"
  echo ">>> Starting stage ${stage} -> ${log}"
  # Stage scripts emit stage-done Telegram via run_round16_notify_helpers.sh
  bash "${script}" 2>&1 | tee -a "${log}"
  echo ">>> Stage ${stage} complete"
}

IFS=',' read -ra _STAGE_LIST <<< "${ROUND16_PIPELINE_STAGES}"
for _raw in "${_STAGE_LIST[@]}"; do
  stage="$(echo "${_raw}" | tr '[:upper:]' '[:lower:]' | xargs)"
  script="${STAGE_SCRIPT[${stage}]:-}"
  if [[ -z "${script}" ]]; then
    echo "ERROR: unknown stage '${stage}' in ROUND16_PIPELINE_STAGES"
    exit 1
  fi
  if [[ ! -f "${script}" ]]; then
    echo "ERROR: missing stage script ${script}"
    exit 1
  fi
  if [[ "${stage}" == "16b" ]]; then
    if [[ ! -f "${ROUND16_ROOT}/reports/round16_top_candidates.csv" ]]; then
      echo "ERROR: missing ${ROUND16_ROOT}/reports/round16_top_candidates.csv (run 16A first)"
      exit 1
    fi
  fi
  run_stage "${stage}" "${script}"
done

CURRENT_STAGE=""
trap - EXIT

echo "========== ROUND16 PIPELINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
notify --event pipeline-done --stages "${ROUND16_PIPELINE_STAGES}"
