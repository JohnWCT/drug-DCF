#!/usr/bin/env bash
# Round 5: generate + pretrain for all three branches (run inside DAPL container).
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
PARALLEL="${PARALLEL:-30}"

for BRANCH in \
  vaewc_round5_control_centered \
  vaewc_round5_class_gap_branch \
  vaewc_round5_t2s_infonce_appendix; do
  RUN_DIR="result/optimization_runs/${BRANCH}"
  echo "=== ${BRANCH} ==="
  python3 tools/optimization_runner.py generate \
    --sweep-spec "config/pretrain_sweeps/${BRANCH}.json" \
    --run-dir "${RUN_DIR}"
  python3 tools/optimization_runner.py pretrain \
    --manifest "${RUN_DIR}/manifests/pretrain_sweep_manifest.csv" \
    --run-dir "${RUN_DIR}" \
    --device "${DEVICE}" \
    --max-parallel "${PARALLEL}"
done

echo "=== Diagnostics ==="
python3 tools/analyze_round5_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round5_control_centered \
    result/optimization_runs/vaewc_round5_class_gap_branch \
    result/optimization_runs/vaewc_round5_t2s_infonce_appendix
