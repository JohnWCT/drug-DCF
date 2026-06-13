#!/usr/bin/env bash
# Resume Round 7 7A pretrain only (after tuning parallel); chain 7B + diagnostics when 7A done.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

LOG="result/optimization_runs/round7_combined/logs/round7_pretrain.log"
mkdir -p result/optimization_runs/round7_combined/logs

_run_branch() {
  local spec="$1"
  local run="result/optimization_runs/${spec}"
  echo "=== [$(date -u +%H:%M:%S)] pretrain ${spec} parallel=${PRETRAIN_PARALLEL} ===" | tee -a "${LOG}"
  bash tools/resume_pretrain_with_profile.sh "${run}" | tee -a "${LOG}"
}

# Reset failed/running for 7A if needed
python3 - <<'PY'
import pandas as pd
p = "result/optimization_runs/vaewc_round7A_exp010_control_refinement/manifests/pretrain_sweep_manifest.csv"
df = pd.read_csv(p)
mask = df["status"].isin(["running", "failed"])
n = int(mask.sum())
if n:
    df.loc[mask, "status"] = "pending"
    df.loc[mask, "error_message"] = ""
    df.to_csv(p, index=False)
    print(f"[round7] reset {n} running/failed -> pending (7A)")
PY

_run_branch "vaewc_round7A_exp010_control_refinement"
_run_branch "vaewc_round7B_vicreg_focused_ablation"

echo "=== [$(date -u +%H:%M:%S)] diagnostics ===" | tee -a "${LOG}"
python3 tools/analyze_round7_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round7A_exp010_control_refinement \
    result/optimization_runs/vaewc_round7B_vicreg_focused_ablation \
  --outdir result/optimization_runs/round7_combined/reports | tee -a "${LOG}"
