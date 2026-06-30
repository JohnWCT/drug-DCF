#!/usr/bin/env bash
set -euo pipefail

ROUND17_ROOT="${ROUND17_ROOT:-result/optimization_runs/round17_direct_proto}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-8}"
DRUG_SMILES_PATH="${DRUG_SMILES_PATH:-data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv}"

echo "========== ROUND17 STAGE 17C START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/round17_direct_proto_config_builder.py \
  --settings config/round17_direct_proto_settings.json \
  --outdir "${ROUND17_ROOT}" \
  --stage 17c \
  --top-candidates "${ROUND17_ROOT}/reports_stage17b/round17_top_candidates.csv" \
  --force

python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND17_ROOT}/manifests/stage17c_finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND17_ROOT}/stage17c" \
  --finetune-config config/params_finetune_round17_direct_proto.json \
  --drug-smiles-path "${DRUG_SMILES_PATH}" \
  --batch-size 12288 \
  --mini-batch-size 3072 \
  --epochs 1500 \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND17_ROOT}/stage17c"

python3 tools/analyze_round17_direct_proto.py \
  --run-dir "${ROUND17_ROOT}/stage17c" \
  --settings config/round17_direct_proto_settings.json \
  --aggregate "${ROUND17_ROOT}/stage17c/aggregate/aggregate_scores.csv" \
  --stage 17c \
  --outdir "${ROUND17_ROOT}/reports_stage17c"

echo "========== ROUND17 STAGE 17C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
