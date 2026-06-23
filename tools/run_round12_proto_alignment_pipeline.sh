#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-33}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-26}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

ROUND12_ROOT="result/optimization_runs/round12_proto_alignment"
ROUND11_ROOT="result/optimization_runs/round11_stability_recon"

echo "========== ROUND12 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round12A] Analyze Round 11 baseline prototype gaps"
python3 tools/analyze_round12_baseline_prototype_gaps.py \
  --round11-root "${ROUND11_ROOT}" \
  --selection "${ROUND11_ROOT}/selection/pretrain_top10.csv" \
  --outdir "${ROUND12_ROOT}/round12a_baseline_qc"

echo "[Round12] Build configs"
python3 tools/round12_config_builder.py \
  --settings config/round12_proto_alignment_settings.json \
  --outdir "${ROUND12_ROOT}" \
  --round11-root "${ROUND11_ROOT}" \
  --force

echo "[Round12] Run pretrain"
python3 tools/optimization_runner.py pretrain \
  --manifest "${ROUND12_ROOT}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${ROUND12_ROOT}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round12] Analyze pretrain"
python3 tools/analyze_round12_proto_alignment.py \
  --run-dir "${ROUND12_ROOT}" \
  --round11-root "${ROUND11_ROOT}" \
  --outdir "${ROUND12_ROOT}/reports"

echo "[Round12] Select candidates"
python3 tools/optimization_runner.py select \
  --run-dir "${ROUND12_ROOT}" \
  --result-dir "${ROUND12_ROOT}/pretrain" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round12_proto_alignment_qc \
  --top-k 30 \
  --min-passing 1 \
  --require-controls 0 \
  --force-baseline-models exp_035 exp_111 \
  --run-tag round12_proto_alignment

echo "[Round12] Finetune"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND12_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND12_ROOT}" \
  --top10 "${ROUND12_ROOT}/selection/pretrain_top10.csv" \
  --finetune-config config/params_finetune_mini.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest

echo "[Round12] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND12_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND12_ROOT}"

echo "[Round12] Final report"
python3 tools/analyze_round12_proto_alignment.py \
  --run-dir "${ROUND12_ROOT}" \
  --round11-root "${ROUND11_ROOT}" \
  --aggregate "${ROUND12_ROOT}/aggregate/aggregate_scores.csv" \
  --selection "${ROUND12_ROOT}/selection/pretrain_top10.csv" \
  --outdir "${ROUND12_ROOT}/final_report"

echo "========== ROUND12 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
