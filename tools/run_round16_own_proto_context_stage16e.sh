#!/usr/bin/env bash
set -euo pipefail

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

echo "========== ROUND16 STAGE 16E START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/round16_bruteforce_config_builder.py \
  --settings config/round16_bruteforce_settings.json \
  --outdir "${ROUND16_ROOT}" \
  --stage 16e \
  --force

python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND16_ROOT}/manifests/stage16e_proto_feature_manifest.csv" \
  --outdir "${ROUND16_ROOT}/features_stage16e"

python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND16_ROOT}/manifests/stage16e_finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND16_ROOT}/stage16e" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND16_ROOT}/stage16e"

python3 tools/analyze_round16_bruteforce.py \
  --run-dir "${ROUND16_ROOT}/stage16e" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round15-root result/optimization_runs/round15_repro_rescue \
  --aggregate "${ROUND16_ROOT}/stage16e/aggregate/aggregate_scores.csv" \
  --stage 16e \
  --outdir "${ROUND16_ROOT}/reports_stage16e"

echo "========== ROUND16 STAGE 16E DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
