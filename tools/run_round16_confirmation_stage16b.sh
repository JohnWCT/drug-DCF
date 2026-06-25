#!/usr/bin/env bash
set -euo pipefail

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"

echo "========== ROUND16 STAGE 16B START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/round16_bruteforce_config_builder.py \
  --settings config/round16_bruteforce_settings.json \
  --outdir "${ROUND16_ROOT}" \
  --stage 16b \
  --top-candidates "${ROUND16_ROOT}/reports/round16_top_candidates.csv" \
  --force

python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND16_ROOT}/manifests/stage16b_proto_feature_manifest.csv" \
  --outdir "${ROUND16_ROOT}/features_stage16b"

python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND16_ROOT}/manifests/stage16b_finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND16_ROOT}/stage16b" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size 12288 \
  --mini-batch-size 3072 \
  --epochs 1500 \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND16_ROOT}/stage16b"

python3 tools/analyze_round16_bruteforce.py \
  --run-dir "${ROUND16_ROOT}/stage16b" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round15-root result/optimization_runs/round15_repro_rescue \
  --aggregate "${ROUND16_ROOT}/stage16b/aggregate/aggregate_scores.csv" \
  --stage 16b \
  --outdir "${ROUND16_ROOT}/final_report"

echo "========== ROUND16 STAGE 16B DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
