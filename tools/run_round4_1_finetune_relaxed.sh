#!/usr/bin/env bash
# Round 4.1: relaxed filter -> Top-N selection -> finetune (+ exp_045 / exp_018 / exp_746 baselines).
set -euo pipefail
cd /workspace/DAPL

RUN_DIR="${RUN_DIR:-result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard}"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
ALL_CAND="${RUN_DIR}/selection/pretrain_all_candidates.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round4_1_finetune_relaxed.log"
FILTER_CONFIG="${FILTER_CONFIG:-config/visualize_vaewc_filter.json}"

FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-26}"
MIN_PASSING="${MIN_PASSING:-6}"
REQUIRE_CONTROLS="${REQUIRE_CONTROLS:-2}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND4.1 RELAXED FILTER + FINETUNE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "filter=${FILTER_CONFIG} (fid<=35, mmd<=0.06, wasserstein<=1.05; kmeans unchanged)"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round4.1 relaxed filter; selection + finetune with integrated TCGA eval."

echo "=== Stage 2: Selection (score_total + relaxed filter) ==="
python3 tools/optimization_runner.py select \
  --run-dir "${RUN_DIR}" \
  --filter-config "${FILTER_CONFIG}" \
  --selection-mode score_total \
  --min-passing "${MIN_PASSING}" \
  --require-controls "${REQUIRE_CONTROLS}" \
  --run-tag round4_1_finetune_relaxed

echo "=== Augment Top-10: exp_045 (best InfoNCE) + exp_018 + exp_746 baselines ==="
python3 tools/augment_finetune_top10.py \
  --top10 "${TOP10}" \
  --all-candidates "${ALL_CAND}" \
  --append-best-infonce-exp045 \
  --append-baseline-exp018 \
  --append-baseline-exp746

echo "=== Stage 3: Finetune manifest ==="
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --top10 "${TOP10}" \
  --build-manifest-only \
  --force-manifest

echo "=== Stage 3: Finetune (parallel=${FINETUNE_MAX_PARALLEL}) ==="
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --top10 "${TOP10}" \
  --epochs 1000 \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH}" \
  --max-parallel "${FINETUNE_MAX_PARALLEL}"

echo "=== Stage 4: Aggregate + report ==="
python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"
python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round4.1 finetune complete (relaxed filter + baselines exp_018/exp_746 + integrated TCGA eval)."

echo "========== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
