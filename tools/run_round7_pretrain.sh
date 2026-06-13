#!/usr/bin/env bash
# Round 7 pretrain (7A control refinement + 7B VICReg ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh
DEVICE="${DEVICE:-cuda}"
PARALLEL="${PARALLEL}"

SKIP_GENERATE="${SKIP_GENERATE:-0}"

for SPEC in \
  vaewc_round7A_exp010_control_refinement \
  vaewc_round7B_vicreg_focused_ablation
do
  RUN="result/optimization_runs/${SPEC}"
  if [[ "${SKIP_GENERATE}" != "1" ]]; then
    echo "=== generate ${SPEC} ==="
    python3 tools/optimization_runner.py generate \
      --sweep-spec "config/pretrain_sweeps/${SPEC}.json" \
      --run-dir "${RUN}" \
      --force
  fi
  echo "=== pretrain ${SPEC} (parallel=${PARALLEL}) ==="
  bash tools/resume_pretrain_with_profile.sh "${RUN}"
done

echo "=== diagnostics ==="
python3 tools/analyze_round7_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round7A_exp010_control_refinement \
    result/optimization_runs/vaewc_round7B_vicreg_focused_ablation \
  --outdir result/optimization_runs/round7_combined/reports
