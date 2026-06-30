#!/usr/bin/env bash
# Resume 16E finetune (higher parallel) then aggregate/analyze/16D.
set -euo pipefail

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
export PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-12}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
MANIFEST="${ROUND16_ROOT}/manifests/stage16e_finetune_dispatch_manifest.csv"
LOG="logs/round16_resume_16e_ed.log"

mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

log "========== RESUME 16E FINETUNE (parallel=${FINETUNE_PARALLEL}) =========="

python3 - <<'PY'
import pandas as pd
path = "result/optimization_runs/round16_bruteforce/manifests/stage16e_finetune_dispatch_manifest.csv"
df = pd.read_csv(path)
mask = df["status"] == "running"
n = int(mask.sum())
if n:
    df.loc[mask, "status"] = "pending"
    df.loc[mask, "start_time"] = ""
    df.to_csv(path, index=False)
    print(f"Reset {n} running jobs -> pending")
else:
    print("No running jobs to reset")
PY

r16_notify --event stage-start --stage 16E

log "16E finetune resume: parallel=${FINETUNE_PARALLEL} batch=${FINETUNE_BATCH_SIZE}"
python3 tools/optimization_runner.py finetune \
  --manifest "${MANIFEST}" \
  --run-dir "${ROUND16_ROOT}/stage16e" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode 2>&1 | tee -a logs/round16_16e_finetune.log

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
log "16E complete"

log "Starting Stage 16D"
bash tools/run_round16_pretrain_vicreg_stage16d.sh 2>&1 | tee -a logs/round16_16d.log
log "16D complete"

r16_notify --event pipeline-done --stages "16f,16e,16d"
log "========== 16E -> 16D DONE =========="
