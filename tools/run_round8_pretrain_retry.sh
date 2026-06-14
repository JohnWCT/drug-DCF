#!/usr/bin/env bash
# Retry Round 8 failed pretrain jobs (CUBLAS/OOM/EmptyDataError) at lower parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

PRETRAIN_PARALLEL="${PRETRAIN_RETRY_PARALLEL:-12}"
RUN8A="result/optimization_runs/vaewc_round8A_control_arch_broad"
RUN8B="result/optimization_runs/vaewc_round8B_vicreg_arch_broad"
LOG="result/optimization_runs/round8_combined/logs/round8_pretrain_retry.log"
mkdir -p result/optimization_runs/round8_combined/logs
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND8 PRETRAIN RETRY $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "parallel=${PRETRAIN_PARALLEL}"

python3 - <<'PY'
import json
import os
import pandas as pd

REPAIRS = [
    (
        "result/optimization_runs/vaewc_round8B_vicreg_arch_broad",
        "exp_proto_015",
        "pretrain/exp_029",
    ),
]

INCOMPLETE_MARKERS = ("gan_metrics.json", "after_traingan_classifier.pth")


def is_complete(exp_dir: str) -> bool:
    return any(os.path.exists(os.path.join(exp_dir, m)) for m in INCOMPLETE_MARKERS)


def cleanup_partial(pretrain_dir: str) -> int:
  removed = 0
  if not os.path.isdir(pretrain_dir):
    return 0
  for name in os.listdir(pretrain_dir):
    if not name.startswith("exp_"):
      continue
    exp_dir = os.path.join(pretrain_dir, name)
    if os.path.isdir(exp_dir) and not is_complete(exp_dir):
      # keep params-only crash artifacts out of the way for a clean retry
      import shutil
      shutil.rmtree(exp_dir)
      removed += 1
      print(f"[cleanup] removed incomplete {exp_dir}")
  return removed


for run_dir, job_id, result_rel in REPAIRS:
    exp_dir = os.path.join(run_dir, result_rel)
    manifest = os.path.join(run_dir, "manifests/pretrain_sweep_manifest.csv")
    if not os.path.exists(manifest):
        continue
    if is_complete(exp_dir):
        df = pd.read_csv(manifest)
        mask = df["job_id"] == job_id
        if mask.any() and df.loc[mask, "status"].iloc[0] != "success":
            df.loc[mask, "status"] = "success"
            df.loc[mask, "error_message"] = ""
            df.loc[mask, "result_dir"] = result_rel
            df.to_csv(manifest, index=False)
            print(f"[repair] marked {job_id} success -> {result_rel}")
    else:
        print(f"[repair] skip {job_id}: {exp_dir} not complete")

for run_dir in (
    "result/optimization_runs/vaewc_round8A_control_arch_broad",
    "result/optimization_runs/vaewc_round8B_vicreg_arch_broad",
):
    n = cleanup_partial(os.path.join(run_dir, "pretrain"))
    print(f"[cleanup] {run_dir}: removed {n} incomplete exp_* dir(s)")

for run_dir in (
    "result/optimization_runs/vaewc_round8A_control_arch_broad",
    "result/optimization_runs/vaewc_round8B_vicreg_arch_broad",
):
    manifest = os.path.join(run_dir, "manifests/pretrain_sweep_manifest.csv")
    df = pd.read_csv(manifest)
    retry = df["status"].isin(["failed", "running"])
    n = int(retry.sum())
    if n:
        df.loc[retry, "status"] = "pending"
        df.loc[retry, "start_time"] = ""
        df.loc[retry, "end_time"] = ""
        df.loc[retry, "error_message"] = ""
        df.to_csv(manifest, index=False)
        print(f"[retry] reset {n} failed/running -> pending in {manifest}")
    print(run_dir, df.status.value_counts().to_dict())
PY

_retry_branch() {
  local run_dir="$1"
  echo "[retry] pretrain ${run_dir} parallel=${PRETRAIN_PARALLEL}"
  python3 tools/optimization_runner.py pretrain \
    --manifest "${run_dir}/manifests/pretrain_sweep_manifest.csv" \
    --run-dir "${run_dir}" \
    --device "${DEVICE:-cuda}" \
    --max-parallel "${PRETRAIN_PARALLEL}"
}

_retry_branch "${RUN8A}"
_retry_branch "${RUN8B}"

for rd in "${RUN8A}" "${RUN8B}"; do
  python3 tools/repair_pretrain_manifest.py --run-dir "${rd}" --force-from-logs
done

python3 tools/analyze_round8_pretrain.py \
  --run-dirs "${RUN8A}" "${RUN8B}" \
  --outdir result/optimization_runs/round8_combined/reports

echo "========== ROUND8 PRETRAIN RETRY DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
