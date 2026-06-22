#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-33}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-26}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

ROUND11_ROOT="result/optimization_runs/round11_stability_recon"

echo "========== ROUND11 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round11A] Post-hoc QC for Round 10 Top-24"
python3 tools/run_round11a_round10_qc.py \
  --round10-root result/optimization_runs/round10_cond_adv \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --outdir "${ROUND11_ROOT}/round11a_qc"

echo "[Round11] Build configs"
python3 tools/round11_config_builder.py \
  --settings config/round11_settings.json \
  --outdir "${ROUND11_ROOT}" \
  --force

echo "[Round11] Run pretrain"
python3 tools/optimization_runner.py pretrain \
  --manifest "${ROUND11_ROOT}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${ROUND11_ROOT}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round11] Analyze pretrain"
python3 tools/analyze_round11_qc.py \
  --run-dir "${ROUND11_ROOT}" \
  --round10-root result/optimization_runs/round10_cond_adv \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --outdir "${ROUND11_ROOT}/reports"

echo "[Round11] Select candidates"
python3 tools/optimization_runner.py select \
  --run-dir "${ROUND11_ROOT}" \
  --result-dir "${ROUND11_ROOT}/pretrain" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round11_stability_qc \
  --top-k 30 \
  --min-passing 1 \
  --require-controls 0 \
  --force-baseline-models exp_111 \
  --run-tag round11_stability_recon

echo "[Round11] Finetune"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND11_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND11_ROOT}" \
  --top10 "${ROUND11_ROOT}/selection/pretrain_top10.csv" \
  --finetune-config config/params_finetune_mini.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest

echo "[Round11] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND11_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND11_ROOT}"

echo "[Round11] Final report"
python3 tools/analyze_round11_qc.py \
  --run-dir "${ROUND11_ROOT}" \
  --round10-root result/optimization_runs/round10_cond_adv \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --aggregate "${ROUND11_ROOT}/aggregate/aggregate_scores.csv" \
  --selection "${ROUND11_ROOT}/selection/pretrain_top10.csv" \
  --outdir "${ROUND11_ROOT}/final_report"

echo "========== ROUND11 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
