#!/usr/bin/env bash
set -euo pipefail

ROUND17R_ROOT="${ROUND17R_ROOT:-result/optimization_runs/round17r_18class}"
SETTINGS="${ROUND17R_SETTINGS:-config/round17r_18class_focused_settings.json}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
DRUG_SMILES_PATH="${DRUG_SMILES_PATH:-data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv}"
TOP_CANDIDATES="${ROUND17R_ROOT}/reports_stage17r_b/round17r_top_candidates.csv"

echo "========== ROUND17R STAGE 17R-C REFINE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

if [[ ! -f "${TOP_CANDIDATES}" ]]; then
  echo "ERROR: missing ${TOP_CANDIDATES} (run 17R-B first)"
  exit 1
fi

python3 tools/round17r_18class_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${ROUND17R_ROOT}" \
  --stage 17r_c \
  --top-candidates "${TOP_CANDIDATES}"

python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND17R_ROOT}/manifests/stage17r_c_finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND17R_ROOT}/stage17r_c" \
  --finetune-config config/params_finetune_round17r_focused.json \
  --drug-smiles-path "${DRUG_SMILES_PATH}" \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND17R_ROOT}/stage17r_c"

python3 tools/analyze_round17r_18class.py \
  --run-dir "${ROUND17R_ROOT}" \
  --settings "${SETTINGS}" \
  --aggregate "${ROUND17R_ROOT}/stage17r_c/aggregate/aggregate_scores.csv" \
  --stage 17r_c \
  --outdir "${ROUND17R_ROOT}/reports_stage17r_c"

echo "========== ROUND17R STAGE 17R-C REFINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
