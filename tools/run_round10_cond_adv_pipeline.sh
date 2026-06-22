#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-33}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-26}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

ROUND10_ROOT="result/optimization_runs/round10_cond_adv"

echo "========== ROUND10 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round10] Build Conditional ADV configs"
python3 tools/round10_config_builder.py \
  --settings config/round10_cond_adv_settings.json \
  --outdir "${ROUND10_ROOT}" \
  --force

echo "[Round10] Run Conditional ADV pretrain"
python3 tools/optimization_runner.py pretrain \
  --manifest "${ROUND10_ROOT}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${ROUND10_ROOT}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round10] Analyze Conditional ADV pretrain"
python3 tools/analyze_round10_cond_adv.py \
  --run-dir "${ROUND10_ROOT}" \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --outdir "${ROUND10_ROOT}/reports"

echo "[Round10] Select candidates"
python3 tools/optimization_runner.py select \
  --run-dir "${ROUND10_ROOT}" \
  --result-dir "${ROUND10_ROOT}/pretrain" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round10_cond_adv_qc \
  --top-k 24 \
  --min-passing 1 \
  --require-controls 0 \
  --force-baseline-models exp_048 \
  --run-tag round10_cond_adv

echo "[Round10] Finetune selected candidates"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND10_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND10_ROOT}" \
  --top10 "${ROUND10_ROOT}/selection/pretrain_top10.csv" \
  --finetune-config config/params_finetune_mini.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest

echo "[Round10] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND10_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND10_ROOT}"

echo "[Round10] Final report"
python3 tools/analyze_round10_cond_adv.py \
  --run-dir "${ROUND10_ROOT}" \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --aggregate "${ROUND10_ROOT}/aggregate/aggregate_scores.csv" \
  --selection "${ROUND10_ROOT}/selection/pretrain_top10.csv" \
  --outdir "${ROUND10_ROOT}/final_report"

echo "========== ROUND10 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
