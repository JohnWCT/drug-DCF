#!/usr/bin/env bash
set -euo pipefail

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

echo "========== ROUND16 STAGE 16F DELTA REPLACEMENT START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/round16_bruteforce_config_builder.py \
  --settings config/round16_bruteforce_settings.json \
  --outdir "${ROUND16_ROOT}" \
  --stage 16f \
  --force

python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND16_ROOT}/manifests/stage16f_proto_feature_manifest.csv" \
  --outdir "${ROUND16_ROOT}/features_stage16f"

python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND16_ROOT}/manifests/stage16f_finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND16_ROOT}/stage16f" \
  --finetune-config config/params_finetune_round16_delta_replacement.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND16_ROOT}/stage16f"

python3 tools/analyze_round16_bruteforce.py \
  --run-dir "${ROUND16_ROOT}/stage16f" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round15-root result/optimization_runs/round15_repro_rescue \
  --aggregate "${ROUND16_ROOT}/stage16f/aggregate/aggregate_scores.csv" \
  --stage 16f \
  --outdir "${ROUND16_ROOT}/reports_stage16f"

echo "========== ROUND16 STAGE 16F DELTA REPLACEMENT DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
