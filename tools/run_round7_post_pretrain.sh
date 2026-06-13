#!/usr/bin/env bash
# Round 7 post-pretrain: diagnostics → diverse selection → finetune → aggregate.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

RUN_DIR="${RUN_DIR:-result/optimization_runs/round7_combined}"
LOG="${RUN_DIR}/logs/round7_post_pretrain.log"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH}"
# Round 7: 120 jobs; use retry parallel (26) to avoid CUBLAS contention at 42.
FINETUNE_MAX_PARALLEL="${FINETUNE_RETRY_PARALLEL:-26}"
SELECTION_TOP_K="${SELECTION_TOP_K:-30}"
SELECTION_MIN_PASSING="${SELECTION_MIN_PASSING:-5}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND7 POST-PRETRAIN $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

RUN7A="result/optimization_runs/vaewc_round7A_exp010_control_refinement"
RUN7B="result/optimization_runs/vaewc_round7B_vicreg_focused_ablation"

echo "=== [diagnostics] ==="
python3 tools/analyze_round7_pretrain.py \
  --run-dirs "${RUN7A}" "${RUN7B}" \
  --outdir "${RUN_DIR}/reports"

RESULT_DIRS_CSV="${RUN7B}/pretrain"

echo "=== [selection] round7_diverse_downstream_probe top_k=${SELECTION_TOP_K} ==="
python3 tools/optimization_runner.py select \
  --run-dir "${RUN_DIR}" \
  --result-dir "${RUN7A}/pretrain" \
  --result-dirs "${RESULT_DIRS_CSV}" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round7_diverse_downstream_probe \
  --exclude-proto-ineffective \
  --force-baseline-models exp_010,exp_012,exp_001,exp_005,exp_746 \
  --top-k "${SELECTION_TOP_K}" \
  --min-passing "${SELECTION_MIN_PASSING}"

TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"

echo "=== [finetune] parallel=${FINETUNE_MAX_PARALLEL} ==="
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --top10 "${TOP10}" \
  --epochs 1000 \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH}" \
  --max-parallel "${FINETUNE_MAX_PARALLEL}"

echo "=== [aggregate] ==="
python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"

echo "=== [report] ==="
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round7 post-pretrain pipeline complete." || true

echo "========== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
