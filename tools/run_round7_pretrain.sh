#!/usr/bin/env bash
# Round 7 pretrain (7A control refinement + 7B VICReg ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
# Match Round 6 pretrain parallelism (~80%+ GPU on typical 24GB nodes).
PARALLEL="${PARALLEL:-20}"

for SPEC in \
  vaewc_round7A_exp010_control_refinement \
  vaewc_round7B_vicreg_focused_ablation
do
  RUN="result/optimization_runs/${SPEC}"
  echo "=== generate ${SPEC} ==="
  python3 tools/optimization_runner.py generate \
    --sweep-spec "config/pretrain_sweeps/${SPEC}.json" \
    --run-dir "${RUN}" \
    --force
  echo "=== pretrain ${SPEC} (parallel=${PARALLEL}) ==="
  python3 tools/optimization_runner.py pretrain \
    --manifest "${RUN}/manifests/pretrain_sweep_manifest.csv" \
    --run-dir "${RUN}" \
    --device "${DEVICE}" \
    --max-parallel "${PARALLEL}"
done

echo "=== diagnostics ==="
python3 tools/analyze_round7_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round7A_exp010_control_refinement \
    result/optimization_runs/vaewc_round7B_vicreg_focused_ablation \
  --outdir result/optimization_runs/round7_combined/reports
