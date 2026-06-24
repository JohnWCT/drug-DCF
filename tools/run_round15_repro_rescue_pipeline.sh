#!/usr/bin/env bash
set -euo pipefail

ROUND15_ROOT="result/optimization_runs/round15_repro_rescue"

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-12}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

echo "========== ROUND15 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round15] Build configs and manifests"
python3 tools/round15_config_builder.py \
  --settings config/round15_repro_rescue_settings.json \
  --outdir "${ROUND15_ROOT}" \
  --force

if [ -f "${ROUND15_ROOT}/manifests/pretrain_sweep_manifest.csv" ]; then
  echo "[Round15] Pretrain ultra-low/late VICReg rescue"
  python3 tools/optimization_runner.py pretrain \
    --manifest "${ROUND15_ROOT}/manifests/pretrain_sweep_manifest.csv" \
    --run-dir "${ROUND15_ROOT}" \
    --batch-size "${PRETRAIN_BATCH_SIZE}" \
    --max-parallel "${PRETRAIN_PARALLEL}"

  echo "[Round15] Analyze and select pretrain rescue candidates"
  python3 tools/analyze_round15_repro_rescue.py \
    --run-dir "${ROUND15_ROOT}" \
    --round13-root result/optimization_runs/round13_proto_response \
    --round14-root result/optimization_runs/round14_vicreg_stabilizer \
    --outdir "${ROUND15_ROOT}/reports"

  python3 tools/optimization_runner.py select \
    --run-dir "${ROUND15_ROOT}" \
    --result-dir "${ROUND15_ROOT}/pretrain" \
    --filter-config config/visualize_vaewc_filter.json \
    --selection-mode round15_repro_rescue_qc \
    --top-k 12 \
    --min-passing 1 \
    --require-controls 0 \
    --run-tag round15_repro_rescue

  echo "[Round15] Rebuild finetune manifest with rescue selection"
  python3 tools/round15_config_builder.py \
    --settings config/round15_repro_rescue_settings.json \
    --outdir "${ROUND15_ROOT}" \
    --build-finetune-manifest \
    --selection "${ROUND15_ROOT}/selection/pretrain_top10.csv" \
    --force
fi

echo "[Round15] Extract compact prototype response features"
python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND15_ROOT}/manifests/proto_feature_manifest.csv" \
  --outdir "${ROUND15_ROOT}/features"

echo "[Round15] Finetune compact features"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND15_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND15_ROOT}" \
  --finetune-config config/params_finetune_round15_compact_features.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

echo "[Round15] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND15_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND15_ROOT}"

echo "[Round15] Final analysis"
python3 tools/analyze_round15_repro_rescue.py \
  --run-dir "${ROUND15_ROOT}" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round14-root result/optimization_runs/round14_vicreg_stabilizer \
  --aggregate "${ROUND15_ROOT}/aggregate/aggregate_scores.csv" \
  --outdir "${ROUND15_ROOT}/final_report"

echo "========== ROUND15 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
