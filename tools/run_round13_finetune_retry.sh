#!/usr/bin/env bash
# Retry failed Round 13 finetune jobs → aggregate → analyze.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-result/optimization_runs/round13_proto_response}"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round13_finetune_retry.log"

FINETUNE_PARALLEL="${FINETUNE_RETRY_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND13 FINETUNE RETRY $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "parallel=${FINETUNE_PARALLEL} batch=${FINETUNE_BATCH_SIZE} mini=${FINETUNE_MINI_BATCH_SIZE}"

python3 - <<PY
import pandas as pd
from pathlib import Path

manifest = Path("${FT_MANIFEST}")
df = pd.read_csv(manifest)
retry = df["status"].isin(["failed", "running"])
n = int(retry.sum())
if n:
    df.loc[retry, "status"] = "pending"
    df.loc[retry, "start_time"] = ""
    df.loc[retry, "end_time"] = ""
    df.loc[retry, "error_message"] = ""
    df.to_csv(manifest, index=False)
    print(f"[retry] reset {n} failed/running job(s) to pending")
print(df["status"].value_counts().to_string())
PY

python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --finetune-config config/params_finetune_proto_features.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --round13-mode

echo "[Round13 retry] Aggregate"
python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"

echo "[Round13 retry] Report"
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"

echo "[Round13 retry] Analyze"
python3 tools/analyze_round13_proto_response.py \
  --run-dir "${RUN_DIR}" \
  --round12-root result/optimization_runs/round12_proto_alignment \
  --round11-root result/optimization_runs/round11_stability_recon \
  --aggregate "${RUN_DIR}/aggregate/aggregate_scores.csv" \
  --outdir "${RUN_DIR}/final_report"

python3 - <<PY
import pandas as pd
df = pd.read_csv("${FT_MANIFEST}")
print("[final manifest]", df["status"].value_counts().to_dict())
PY

echo "========== ROUND13 FINETUNE RETRY DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
