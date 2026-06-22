#!/usr/bin/env bash
set -euo pipefail

R="${ROUND11_ROOT:-result/optimization_runs/round11_stability_recon}"

echo "[Analyze] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 tools/analyze_round11_qc.py \
  --run-dir "${R}" \
  --round10-root result/optimization_runs/round10_cond_adv \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --outdir "${R}/reports"

echo "[Select] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 tools/optimization_runner.py select \
  --run-dir "${R}" \
  --result-dir "${R}/pretrain" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round11_stability_qc \
  --top-k 30 \
  --min-passing 1 \
  --require-controls 0 \
  --force-baseline-models exp_111 \
  --run-tag round11_stability_recon

echo "[Finetune] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 tools/optimization_runner.py finetune \
  --manifest "${R}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${R}" \
  --top10 "${R}/selection/pretrain_top10.csv" \
  --finetune-config config/params_finetune_mini.json \
  --batch-size "${FINETUNE_BATCH_SIZE:-12288}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE:-3072}" \
  --epochs "${FINETUNE_EPOCHS:-1000}" \
  --max-parallel "${FINETUNE_PARALLEL:-26}" \
  --force-manifest

echo "[Aggregate] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 tools/optimization_runner.py aggregate --run-dir "${R}"
python3 tools/optimization_runner.py report --run-dir "${R}"

echo "[Final] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 tools/analyze_round11_qc.py \
  --run-dir "${R}" \
  --round10-root result/optimization_runs/round10_cond_adv \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --aggregate "${R}/aggregate/aggregate_scores.csv" \
  --selection "${R}/selection/pretrain_top10.csv" \
  --outdir "${R}/final_report"

echo "========== ROUND11 DOWNSTREAM DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
