#!/usr/bin/env bash
# Retry failed 16E finetune jobs, then re-aggregate and analyze.
set -euo pipefail

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
MANIFEST="${ROUND16_ROOT}/manifests/stage16e_finetune_dispatch_manifest.csv"
LOG="logs/round16_retry_16e_failed.log"

mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

log "========== RETRY 16E FAILED (parallel=${FINETUNE_PARALLEL}) =========="

python3 - <<'PY'
import pandas as pd
path = "result/optimization_runs/round16_bruteforce/manifests/stage16e_finetune_dispatch_manifest.csv"
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
vc = df["status"].value_counts().to_dict()
print(vc)
PY

r16_notify --event stage-start --stage 16E

log "16E finetune retry: parallel=${FINETUNE_PARALLEL} batch=${FINETUNE_BATCH_SIZE}"
python3 tools/optimization_runner.py finetune \
  --manifest "${MANIFEST}" \
  --run-dir "${ROUND16_ROOT}/stage16e" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode 2>&1 | tee -a logs/round16_16e_finetune_retry.log

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND16_ROOT}/stage16e"

python3 tools/analyze_round16_bruteforce.py \
  --run-dir "${ROUND16_ROOT}/stage16e" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round15-root result/optimization_runs/round15_repro_rescue \
  --aggregate "${ROUND16_ROOT}/stage16e/aggregate/aggregate_scores.csv" \
  --stage 16e \
  --outdir "${ROUND16_ROOT}/reports_stage16e"

r16_notify --event stage-done --stage 16E
log "========== 16E RETRY DONE =========="
