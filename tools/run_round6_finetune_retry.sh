#!/usr/bin/env bash
# Retry failed Round 6 finetune jobs → aggregate → report.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

RUN_DIR="${RUN_DIR:-result/optimization_runs/round6_combined}"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round6_finetune_retry.log"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND6 FINETUNE RETRY $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "parallel=${FINETUNE_MAX_PARALLEL} batch=${FINETUNE_BATCH_SIZE} mini=${FINETUNE_MINI_BATCH}"

python3 - <<'PY'
import pandas as pd
from pathlib import Path

manifest = Path("result/optimization_runs/round6_combined/manifests/finetune_dispatch_manifest.csv")
df = pd.read_csv(manifest)
retry = df["status"].isin(["failed", "running"])
n = int(retry.sum())
if n == 0:
    print("[retry] no failed/running jobs in manifest")
else:
    df.loc[retry, "status"] = "pending"
    df.loc[retry, "start_time"] = ""
    df.loc[retry, "end_time"] = ""
    df.loc[retry, "error_message"] = ""
    df.to_csv(manifest, index=False)
    print(f"[retry] reset {n} failed/running job(s) to pending")
print(df["status"].value_counts().to_string())
PY

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round6 finetune retry (max_parallel=${FINETUNE_MAX_PARALLEL})." || true

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
  --note "Round6 finetune retry complete." || true

python3 - <<'PY'
import pandas as pd
manifest = "result/optimization_runs/round6_combined/manifests/finetune_dispatch_manifest.csv"
df = pd.read_csv(manifest)
print("[final manifest]", df["status"].value_counts().to_dict())
PY

echo "========== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
