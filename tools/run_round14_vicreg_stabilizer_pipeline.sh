#!/usr/bin/env bash
set -euo pipefail

ROUND14_ROOT="result/optimization_runs/round14_vicreg_stabilizer"
ROUND13_ROOT="result/optimization_runs/round13_proto_response"
ROUND12_ROOT="result/optimization_runs/round12_proto_alignment"

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-20}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

echo "========== ROUND14 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round14] Build VICReg configs"
python3 tools/round14_config_builder.py \
  --settings config/round14_vicreg_stabilizer_settings.json \
  --outdir "${ROUND14_ROOT}" \
  --force

echo "[Round14] Reference controls (Round 13 baselines)"
python3 tools/round14_config_builder.py \
  --settings config/round14_vicreg_stabilizer_settings.json \
  --outdir "${ROUND14_ROOT}" \
  --build-reference-controls

echo "[Round14] Pretrain"
python3 tools/optimization_runner.py pretrain \
  --manifest "${ROUND14_ROOT}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${ROUND14_ROOT}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round14] Analyze pretrain and select candidates"
python3 tools/analyze_round14_vicreg_stabilizer.py \
  --run-dir "${ROUND14_ROOT}" \
  --round13-root "${ROUND13_ROOT}" \
  --round12-root "${ROUND12_ROOT}" \
  --outdir "${ROUND14_ROOT}/reports"

python3 tools/optimization_runner.py select \
  --run-dir "${ROUND14_ROOT}" \
  --result-dir "${ROUND14_ROOT}/pretrain" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round14_vicreg_stabilizer_qc \
  --top-k 16 \
  --min-passing 1 \
  --require-controls 0 \
  --run-tag round14_vicreg_stabilizer

echo "[Round14] Build response-feature manifests"
python3 tools/round14_config_builder.py \
  --settings config/round14_vicreg_stabilizer_settings.json \
  --outdir "${ROUND14_ROOT}" \
  --build-finetune-manifest \
  --selection "${ROUND14_ROOT}/selection/pretrain_top10.csv" \
  --force

echo "[Round14] Extract compact prototype response features"
python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND14_ROOT}/manifests/proto_feature_manifest.csv" \
  --outdir "${ROUND14_ROOT}/features"

echo "[Round14] Finetune"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND14_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND14_ROOT}" \
  --finetune-config config/params_finetune_round14_proto_features.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

echo "[Round14] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND14_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND14_ROOT}"

echo "[Round14] Final analysis"
python3 tools/analyze_round14_vicreg_stabilizer.py \
  --run-dir "${ROUND14_ROOT}" \
  --round13-root "${ROUND13_ROOT}" \
  --round12-root "${ROUND12_ROOT}" \
  --aggregate "${ROUND14_ROOT}/aggregate/aggregate_scores.csv" \
  --outdir "${ROUND14_ROOT}/final_report"

echo "========== ROUND14 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
