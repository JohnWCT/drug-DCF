#!/usr/bin/env bash
# Round 17 direct-prototype pipeline: Stage 17A → 17B → 17C with Telegram notifications.
set -euo pipefail

export ROUND17_ROOT="${ROUND17_ROOT:-result/optimization_runs/round17_direct_proto}"
export ROUND17_PIPELINE_STAGES="${ROUND17_PIPELINE_STAGES:-17a,17b,17c}"
export DRUG_SMILES_PATH="${DRUG_SMILES_PATH:-data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv}"

# High GPU utilization (~85%+ SM on RTX 6000 Ada): larger batch + higher parallel vs defaults.
export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-26}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
export ROUND17_RERUN_COMPLETED="${ROUND17_RERUN_COMPLETED:-0}"

LOG_DIR="${LOG_DIR:-logs}"
CURRENT_STAGE=""
mkdir -p "${LOG_DIR}"

declare -A STAGE_SCRIPT=(
  [17a]="tools/run_round17_direct_proto_stage17a.sh"
  [17b]="tools/run_round17_proto_head_stage17b.sh"
  [17c]="tools/run_round17_confirmation_stage17c.sh"
)

notify() {
  python3 tools/round17_telegram_notify.py "$@" || true
}

on_exit() {
  local code=$?
  if [[ ${code} -ne 0 && -n "${CURRENT_STAGE}" ]]; then
    notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code ${code}"
    notify --event pipeline-fail --reason "stage ${CURRENT_STAGE} failed (exit ${code})"
  fi
}
trap on_exit EXIT

echo "========== ROUND17 PIPELINE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "ROUND17_ROOT=${ROUND17_ROOT}"
echo "ROUND17_PIPELINE_STAGES=${ROUND17_PIPELINE_STAGES}"
echo "FINETUNE_PARALLEL=${FINETUNE_PARALLEL}"
echo "FINETUNE_BATCH_SIZE=${FINETUNE_BATCH_SIZE}"
echo "FINETUNE_MINI_BATCH_SIZE=${FINETUNE_MINI_BATCH_SIZE}"
echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"

notify --event pipeline-start --stages "${ROUND17_PIPELINE_STAGES}"

run_stage() {
  local stage="$1"
  local script="$2"
  local log="${LOG_DIR}/round17_${stage}.log"
  CURRENT_STAGE="${stage}"
  echo ">>> Starting stage ${stage} -> ${log}"
  bash "${script}" 2>&1 | tee -a "${log}"
  echo ">>> Stage ${stage} complete"
}

IFS=',' read -ra _STAGE_LIST <<< "${ROUND17_PIPELINE_STAGES}"
for _raw in "${_STAGE_LIST[@]}"; do
  stage="$(echo "${_raw}" | tr '[:upper:]' '[:lower:]' | xargs)"
  script="${STAGE_SCRIPT[${stage}]:-}"
  if [[ -z "${script}" ]]; then
    echo "ERROR: unknown stage '${stage}' in ROUND17_PIPELINE_STAGES"
    exit 1
  fi
  if [[ ! -f "${script}" ]]; then
    echo "ERROR: missing stage script ${script}"
    exit 1
  fi
  if [[ "${stage}" == "17b" ]]; then
    if [[ ! -f "${ROUND17_ROOT}/reports_stage17a/round17_top_candidates.csv" ]]; then
      echo "ERROR: missing ${ROUND17_ROOT}/reports_stage17a/round17_top_candidates.csv (run 17A first)"
      exit 1
    fi
  fi
  if [[ "${stage}" == "17c" ]]; then
    if [[ ! -f "${ROUND17_ROOT}/reports_stage17b/round17_top_candidates.csv" ]]; then
      echo "ERROR: missing ${ROUND17_ROOT}/reports_stage17b/round17_top_candidates.csv (run 17B first)"
      exit 1
    fi
  fi
  run_stage "${stage}" "${script}"
done

CURRENT_STAGE=""
trap - EXIT

echo "========== ROUND17 PIPELINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
notify --event pipeline-done --stages "${ROUND17_PIPELINE_STAGES}"
