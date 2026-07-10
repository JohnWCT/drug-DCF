#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=tools/run_round17_notify_helpers.sh
source "$(dirname "$0")/run_round17_notify_helpers.sh"

ROUND17R_ROOT="${ROUND17R_ROOT:-result/optimization_runs/round17r_18class}"
SETTINGS="${ROUND17R_SETTINGS:-config/round17r_18class_focused_settings.json}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
DRUG_SMILES_PATH="${DRUG_SMILES_PATH:-data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv}"
FT_MANIFEST="${ROUND17R_ROOT}/manifests/stage17r_b_finetune_dispatch_manifest.csv"

CURRENT_STAGE="17R-B"
on_exit() {
  local code=$?
  if [[ ${code} -ne 0 ]]; then
    r17_notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code ${code}" \
      --manifest "${FT_MANIFEST}" || true
  fi
}
trap on_exit EXIT

echo "========== ROUND17R STAGE 17R-B FOCUSED START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "FINETUNE_PARALLEL=${FINETUNE_PARALLEL}"
r17_notify --event stage-start --stage 17R-B --manifest "${FT_MANIFEST}"

python3 tools/round17r_18class_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${ROUND17R_ROOT}" \
  --stage 17r_b

python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND17R_ROOT}/manifests/stage17r_b_proto_feature_manifest.csv" \
  --outdir "${ROUND17R_ROOT}/features" \
  --strict

python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${ROUND17R_ROOT}/stage17r_b" \
  --finetune-config config/params_finetune_round17r_focused.json \
  --drug-smiles-path "${DRUG_SMILES_PATH}" \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND17R_ROOT}/stage17r_b"

python3 tools/analyze_round17r_18class.py \
  --run-dir "${ROUND17R_ROOT}" \
  --settings "${SETTINGS}" \
  --aggregate "${ROUND17R_ROOT}/stage17r_b/aggregate/aggregate_scores.csv" \
  --stage 17r_b \
  --outdir "${ROUND17R_ROOT}/reports_stage17r_b"

r17_notify --event stage-done --stage 17R-B --manifest "${FT_MANIFEST}"
trap - EXIT
echo "========== ROUND17R STAGE 17R-B FOCUSED DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
