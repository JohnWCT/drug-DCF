#!/usr/bin/env bash
set -euo pipefail

ROUND13_ROOT="result/optimization_runs/round13_proto_response"

FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-26}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

echo "========== ROUND13 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round13] Build configs and manifests"
python3 tools/round13_config_builder.py \
  --settings config/round13_proto_response_settings.json \
  --outdir "${ROUND13_ROOT}" \
  --force

echo "[Round13] Extract prototype response features"
python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND13_ROOT}/manifests/proto_feature_manifest.csv" \
  --outdir "${ROUND13_ROOT}/features"

echo "[Round13] Run finetune"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND13_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND13_ROOT}" \
  --finetune-config config/params_finetune_proto_features.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

echo "[Round13] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND13_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND13_ROOT}"

echo "[Round13] Analyze"
python3 tools/analyze_round13_proto_response.py \
  --run-dir "${ROUND13_ROOT}" \
  --round12-root result/optimization_runs/round12_proto_alignment \
  --round11-root result/optimization_runs/round11_stability_recon \
  --aggregate "${ROUND13_ROOT}/aggregate/aggregate_scores.csv" \
  --outdir "${ROUND13_ROOT}/final_report"

echo "========== ROUND13 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
