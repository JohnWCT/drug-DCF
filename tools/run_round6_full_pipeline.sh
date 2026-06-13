#!/usr/bin/env bash
# Round 6 full pipeline: pretrain (6A–6E) → diagnostics → selection → finetune → aggregate → report.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-result/optimization_runs/round6_combined}"
LOG="${RUN_DIR}/logs/round6_full_pipeline.log"
DEVICE="${DEVICE:-cuda}"
PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-20}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-42}"
SELECTION_TOP_K="${SELECTION_TOP_K:-30}"
SELECTION_MIN_PASSING="${SELECTION_MIN_PASSING:-5}"

BRANCHES=(
  vaewc_round6A_tumor_topology
  vaewc_round6B_topology_classgap_combo
  vaewc_round6C_tumor_transfer_subspace
  vaewc_round6D_within_domain_tumor_supcon
  vaewc_round6E_tumor_vicreg_stabilizer
)

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND6 FULL PIPELINE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "pretrain_parallel=${PRETRAIN_PARALLEL} finetune_parallel=${FINETUNE_MAX_PARALLEL} top_k=${SELECTION_TOP_K}"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round6 full pipeline started (pretrain_parallel=${PRETRAIN_PARALLEL})." || true

for SPEC in "${BRANCHES[@]}"; do
  BRANCH_RUN="result/optimization_runs/${SPEC}"
  echo "=== [pretrain] generate ${SPEC} ==="
  python3 tools/optimization_runner.py generate \
    --sweep-spec "config/pretrain_sweeps/${SPEC}.json" \
    --run-dir "${BRANCH_RUN}" \
    --force
  echo "=== [pretrain] ${SPEC} max_parallel=${PRETRAIN_PARALLEL} ==="
  python3 tools/optimization_runner.py pretrain \
    --manifest "${BRANCH_RUN}/manifests/pretrain_sweep_manifest.csv" \
    --run-dir "${BRANCH_RUN}" \
    --device "${DEVICE}" \
    --max-parallel "${PRETRAIN_PARALLEL}"
done

echo "=== [diagnostics] ==="
python3 tools/analyze_round6_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round6A_tumor_topology \
    result/optimization_runs/vaewc_round6B_topology_classgap_combo \
    result/optimization_runs/vaewc_round6C_tumor_transfer_subspace \
    result/optimization_runs/vaewc_round6D_within_domain_tumor_supcon \
    result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer \
  --out-dir "${RUN_DIR}/reports"

RESULT_DIRS=(
  result/optimization_runs/vaewc_round6B_topology_classgap_combo/pretrain
  result/optimization_runs/vaewc_round6C_tumor_transfer_subspace/pretrain
  result/optimization_runs/vaewc_round6D_within_domain_tumor_supcon/pretrain
  result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer/pretrain
)
RESULT_DIRS_CSV=$(IFS=,; echo "${RESULT_DIRS[*]}")

echo "=== [selection] round6_sweetspot top_k=${SELECTION_TOP_K} ==="
python3 tools/optimization_runner.py select \
  --run-dir "${RUN_DIR}" \
  --result-dir result/optimization_runs/vaewc_round6A_tumor_topology/pretrain \
  --result-dirs "${RESULT_DIRS_CSV}" \
  --selection-mode round6_sweetspot \
  --exclude-proto-ineffective \
  --force-baseline-models exp_001,exp_005,exp_746 \
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
  --note "Round6 full pipeline complete." || true

echo "========== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
