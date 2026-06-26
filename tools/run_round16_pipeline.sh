#!/usr/bin/env bash
# Round 16 full pipeline: 16F -> 16E -> 16A -> 16B -> 16C (16D skipped: not implemented)
set -euo pipefail

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-16}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
LOG_DIR="logs"
CURRENT_STAGE=""

mkdir -p "${LOG_DIR}"

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
echo "FINETUNE_PARALLEL=${FINETUNE_PARALLEL}"
echo "FINETUNE_BATCH_SIZE=${FINETUNE_BATCH_SIZE}"
echo "FINETUNE_MINI_BATCH_SIZE=${FINETUNE_MINI_BATCH_SIZE}"
echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"

notify --event pipeline-start

run_stage() {
  local stage="$1"
  local script="$2"
  local log="${LOG_DIR}/round16_${stage}.log"
  CURRENT_STAGE="${stage}"
  echo ">>> Starting stage ${stage} -> ${log}"
  notify --event stage-start --stage "${stage}"
  bash "${script}" 2>&1 | tee -a "${log}"
  notify --event stage-done --stage "${stage}"
  echo ">>> Stage ${stage} complete"
}

run_stage "16F" "tools/run_round16_delta_replacement_stage16f.sh"
run_stage "16E" "tools/run_round16_own_proto_context_stage16e.sh"
run_stage "16A" "tools/run_round16_bruteforce_stage16a.sh"

if [[ ! -f "${ROUND16_ROOT}/reports/round16_top_candidates.csv" ]]; then
  echo "ERROR: missing ${ROUND16_ROOT}/reports/round16_top_candidates.csv after 16A"
  exit 1
fi

run_stage "16B" "tools/run_round16_confirmation_stage16b.sh"
run_stage "16C" "tools/run_round16_feature_variants_stage16c.sh"

CURRENT_STAGE=""
trap - EXIT

echo "========== ROUND16 PIPELINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "16D skipped (disabled / not implemented in config)"
notify --event pipeline-done
