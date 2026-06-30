#!/usr/bin/env bash
# Wait for Stage 16F to finish, then run 16E -> 16D (architecture round).
set -euo pipefail

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
export PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-12}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
MANIFEST="${ROUND16_ROOT}/manifests/stage16f_finetune_dispatch_manifest.csv"
LOG="logs/round16_after_16f_ed.log"
EXPECTED_F=384
POLL_SEC="${POLL_SEC:-120}"

mkdir -p logs

notify() {
  python3 tools/round16_telegram_notify.py "$@" || true
}

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

f_done() {
  python3 - <<'PY'
import pandas as pd
import sys
df = pd.read_csv("result/optimization_runs/round16_bruteforce/manifests/stage16f_finetune_dispatch_manifest.csv")
vc = df["status"].value_counts().to_dict()
ok = int(vc.get("success", 0))
fail = int(vc.get("failed", 0))
pending = int(vc.get("pending", 0))
running = int(vc.get("running", 0))
total = len(df)
print(f"{ok}/{total} success, {fail} failed, pending={pending}, running={running}")
sys.exit(0 if pending == 0 and running == 0 else 1)
PY
}

log "========== WAIT 16F -> RUN 16E -> 16D =========="

while true; do
  if f_done; then
    status_line="$(python3 - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/round16_bruteforce/manifests/stage16f_finetune_dispatch_manifest.csv")
vc = df["status"].value_counts().to_dict()
print(f"success={vc.get('success',0)} failed={vc.get('failed',0)} running={vc.get('running',0)} pending={vc.get('pending',0)}")
PY
)"
    log "16F complete: ${status_line}"
    break
  fi
  status_line="$(python3 - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/round16_bruteforce/manifests/stage16f_finetune_dispatch_manifest.csv")
vc = df["status"].value_counts().to_dict()
print(f"success={vc.get('success',0)} running={vc.get('running',0)} pending={vc.get('pending',0)}")
PY
)" || status_line="manifest unreadable"
  log "16F in progress (${status_line}); sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

notify --event stage-done --stage "16F"

if pgrep -f "bash tools/run_round16_pipeline.sh" >/dev/null; then
  log "Stopping legacy pipeline orchestrator before 16E"
  pkill -f "bash tools/run_round16_pipeline.sh" || true
  sleep 3
fi
pkill -f "bash tools/run_round16_own_proto_context_stage16e.sh" 2>/dev/null || true
sleep 2

log "Starting Stage 16E"
bash tools/run_round16_own_proto_context_stage16e.sh 2>&1 | tee -a logs/round16_16e.log
log "16E complete"

log "Starting Stage 16D"
bash tools/run_round16_pretrain_vicreg_stage16d.sh 2>&1 | tee -a logs/round16_16d.log
log "16D complete"

notify --event pipeline-done --stages "16f,16e,16d"
log "========== 16F -> 16E -> 16D DONE =========="
