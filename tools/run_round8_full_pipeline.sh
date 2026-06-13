#!/usr/bin/env bash
# Round 8 full pipeline: 8A+8B pretrain → selection → first-pass finetune → sensitivity finetune.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-33}"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-${FINETUNE_MAX_PARALLEL:-26}}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"
SELECTION_TOP_K="${SELECTION_TOP_K:-50}"
SELECTION_MIN_PASSING="${SELECTION_MIN_PASSING:-1}"

RUN8A="result/optimization_runs/vaewc_round8A_control_arch_broad"
RUN8B="result/optimization_runs/vaewc_round8B_vicreg_arch_broad"
RUN8C="result/optimization_runs/round8_combined"
RUN8D="result/optimization_runs/round8_finetune_sensitivity"
LOG="${RUN8C}/logs/round8_full_pipeline.log"

mkdir -p "${RUN8C}/logs" "${RUN8D}/selection" "${RUN8D}/manifests"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND8 FULL PIPELINE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "pretrain_parallel=${PRETRAIN_PARALLEL} finetune_parallel=${FINETUNE_PARALLEL} top_k=${SELECTION_TOP_K}"

echo "[Round8] Generate 8A configs"
python3 tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round8A_control_arch_broad.json \
  --run-dir "${RUN8A}" \
  --force

echo "[Round8] Generate 8B configs"
python3 tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round8B_vicreg_arch_broad.json \
  --run-dir "${RUN8B}" \
  --force

echo "[Round8] Pretrain 8A"
python3 tools/optimization_runner.py pretrain \
  --manifest "${RUN8A}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${RUN8A}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round8] Pretrain 8B"
python3 tools/optimization_runner.py pretrain \
  --manifest "${RUN8B}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${RUN8B}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

echo "[Round8] Pretrain diagnostics"
python3 tools/analyze_round8_pretrain.py \
  --run-dirs "${RUN8A}" "${RUN8B}" \
  --outdir "${RUN8C}/reports"

echo "[Round8] Combined selection top_k=${SELECTION_TOP_K}"
python3 tools/optimization_runner.py select \
  --run-dir "${RUN8C}" \
  --result-dir "${RUN8A}/pretrain" \
  --result-dirs "${RUN8B}/pretrain" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round8_architecture_broad_probe \
  --exclude-proto-ineffective \
  --force-baseline-models exp_048,exp_021,exp_010,exp_012,exp_005,exp_746 \
  --top-k "${SELECTION_TOP_K}" \
  --min-passing "${SELECTION_MIN_PASSING}" \
  --require-controls 0

echo "[Round8] First-pass finetune"
python3 tools/optimization_runner.py finetune \
  --manifest "${RUN8C}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${RUN8C}" \
  --top10 "${RUN8C}/selection/pretrain_top10.csv" \
  --finetune-config config/params_finetune_mini.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest

echo "[Round8] First-pass aggregate"
python3 tools/optimization_runner.py aggregate --run-dir "${RUN8C}"
python3 tools/optimization_runner.py report --run-dir "${RUN8C}"

echo "[Round8] Build second-pass sensitivity model_select"
python3 tools/build_round8_finetune_sensitivity_select.py \
  --aggregate "${RUN8C}/aggregate/aggregate_scores.csv" \
  --selection "${RUN8C}/selection/pretrain_top10.csv" \
  --outdir "${RUN8D}/selection" \
  --max-models 12 \
  --force-models exp_048,exp_021,exp_746

echo "[Round8] Second-pass finetune sensitivity"
python3 tools/optimization_runner.py finetune \
  --manifest "${RUN8D}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${RUN8D}" \
  --top10 "${RUN8D}/selection/model_select.csv" \
  --finetune-config config/finetune_sweeps/round8_finetune_sensitivity_broad.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest

echo "[Round8] Second-pass aggregate"
python3 tools/optimization_runner.py aggregate --run-dir "${RUN8D}"
python3 tools/optimization_runner.py report --run-dir "${RUN8D}"

python3 tools/update_running_report.py --run-dir "${RUN8C}" \
  --note "Round8 full pipeline complete (first-pass)." || true
python3 tools/update_running_report.py --run-dir "${RUN8D}" \
  --note "Round8 sensitivity pass complete." || true

echo "========== ROUND8 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
