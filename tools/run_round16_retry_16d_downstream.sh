#!/usr/bin/env bash
# Retry failed Stage 16D downstream finetune only (filter + features assumed done).
set -euo pipefail

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
STAGE_ROOT="${ROUND16_ROOT}/stage16d"
DOWNSTREAM_RUN="${STAGE_ROOT}/downstream"
MANIFEST="${STAGE_ROOT}/manifests/stage16d_downstream_finetune_manifest.csv"
LOG="logs/round16_retry_16d_downstream.log"

mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

log "========== RETRY 16D DOWNSTREAM (parallel=${FINETUNE_PARALLEL}) =========="

python3 - <<'PY'
import pandas as pd

path = "result/optimization_runs/round16_bruteforce/stage16d/manifests/stage16d_downstream_finetune_manifest.csv"
df = pd.read_csv(path)
for status in ("failed", "running"):
    mask = df["status"] == status
    n = int(mask.sum())
    if n:
        df.loc[mask, "status"] = "pending"
        df.loc[mask, "start_time"] = ""
        df.loc[mask, "end_time"] = ""
        df.loc[mask, "error_message"] = ""
        print(f"Reset {n} {status} -> pending")
df.to_csv(path, index=False)
print(df["status"].value_counts().to_dict())
PY

r16_notify --event stage-start --stage 16D

log "16D downstream finetune retry: parallel=${FINETUNE_PARALLEL} batch=${FINETUNE_BATCH_SIZE}"
python3 tools/optimization_runner.py finetune \
  --manifest "${MANIFEST}" \
  --run-dir "${DOWNSTREAM_RUN}" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode 2>&1 | tee -a logs/round16_16d_downstream_finetune_retry.log

python3 tools/optimization_runner.py aggregate \
  --run-dir "${DOWNSTREAM_RUN}"

python3 tools/analyze_round16_stage16d.py \
  --stage-root "${STAGE_ROOT}" \
  --outdir "${STAGE_ROOT}/reports"

r16_notify --event stage-done --stage 16D
log "========== 16D DOWNSTREAM RETRY DONE =========="
