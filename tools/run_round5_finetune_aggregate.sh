#!/usr/bin/env bash
# Round 5 downstream: finetune (resume) → aggregate → report.
# Tuned for RTX 6000 Ada (49GB): higher max_parallel when GPU util < 80%.
set -euo pipefail
cd /workspace/DAPL

RUN_DIR="${RUN_DIR:-result/optimization_runs/round5_combined}"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round5_finetune_aggregate.log"

# 26 parallel ≈ 50% GPU on RTX 6000 Ada; 42 targets ~80% (adjust via env if OOM).
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-42}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND5 FINETUNE+AGGREGATE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "parallel=${FINETUNE_MAX_PARALLEL} batch=${FINETUNE_BATCH_SIZE} mini=${FINETUNE_MINI_BATCH}"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round5 finetune resume (max_parallel=${FINETUNE_MAX_PARALLEL})."

python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --top10 "${TOP10}" \
  --epochs 1000 \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH}" \
  --max-parallel "${FINETUNE_MAX_PARALLEL}"

echo "=== Aggregate ==="
python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"

echo "=== Report ==="
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round5 finetune+aggregate complete (max_parallel=${FINETUNE_MAX_PARALLEL})."

echo "========== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
