#!/usr/bin/env bash
# Round 9: deconfounding QC, baseline reproduction, conditional diagnostics, mini finetune.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-20}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-26}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

ROUND9_ROOT="result/optimization_runs/round9_diagnostics"
ROUND9_REPRO="result/optimization_runs/round9_reproduction"
LOG="${ROUND9_ROOT}/logs/round9_diagnostics_pipeline.log"
mkdir -p "${ROUND9_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND9 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[Round9] Resolve baseline checkpoints"
python3 tools/round9_baseline_resolver.py \
  --baseline-config config/round9_baselines.json \
  --search-root result \
  --outdir "${ROUND9_ROOT}/baselines"

echo "[Round9] Build reproduction manifest"
python3 tools/build_round9_reproduction_manifest.py \
  --resolved-baselines "${ROUND9_ROOT}/baselines/resolved_baselines.csv" \
  --baseline-config config/round9_baselines.json \
  --outdir "${ROUND9_REPRO}" \
  --force

echo "[Round9] Run reproduction pretrain"
python3 tools/optimization_runner.py pretrain \
  --manifest "${ROUND9_REPRO}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${ROUND9_REPRO}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round9] Deconfounding QC"
python3 tools/analyze_deconfounding_qc.py \
  --run-dir "${ROUND9_REPRO}" \
  --latent-view shared \
  --min-source-per-cancer 10 \
  --min-target-per-cancer 10 \
  --classifiers logistic_regression small_mlp \
  --include-fid \
  --include-wasserstein \
  --include-clustering \
  --outdir "${ROUND9_ROOT}/reports"

echo "[Round9] Conditional domain leakage diagnostics"
python3 tools/analyze_conditional_domain_leakage.py \
  --run-dir "${ROUND9_REPRO}" \
  --latent-view shared \
  --min-source-per-cancer 10 \
  --min-target-per-cancer 10 \
  --classifiers logistic_regression small_mlp \
  --outdir "${ROUND9_ROOT}/reports"

echo "[Round9] Cancer prototype diagnostics"
python3 tools/analyze_cancer_prototypes.py \
  --run-dir "${ROUND9_REPRO}" \
  --latent-view shared \
  --min-source-per-cancer 10 \
  --min-target-per-cancer 10 \
  --metrics cosine euclidean \
  --outdir "${ROUND9_ROOT}/reports"

echo "[Round9] Latent stability diagnostics"
python3 tools/analyze_latent_stability.py \
  --run-dir "${ROUND9_REPRO}" \
  --latent-view shared \
  --outdir "${ROUND9_ROOT}/reports"

echo "[Round9] Build finetune model_select"
python3 tools/build_round9_finetune_select.py \
  --run-dir "${ROUND9_REPRO}" \
  --resolved-baselines "${ROUND9_ROOT}/baselines/resolved_baselines.csv" \
  --outdir "${ROUND9_ROOT}/selection" \
  --include-all-reproductions

echo "[Round9] Finetune reproduction checkpoints"
python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND9_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND9_ROOT}" \
  --top10 "${ROUND9_ROOT}/selection/model_select.csv" \
  --finetune-config config/params_finetune_mini.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest

echo "[Round9] Aggregate finetune scores"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND9_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND9_ROOT}"

echo "[Round9] Final diagnostics report"
python3 tools/analyze_round9_diagnostics.py \
  --diagnostics-dir "${ROUND9_ROOT}/reports" \
  --aggregate "${ROUND9_ROOT}/aggregate/aggregate_scores.csv" \
  --resolved-baselines "${ROUND9_ROOT}/baselines/resolved_baselines.csv" \
  --outdir "${ROUND9_ROOT}/final_report"

echo "========== ROUND9 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
