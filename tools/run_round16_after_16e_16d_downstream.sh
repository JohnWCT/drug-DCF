#!/usr/bin/env bash
# Wait for the running 16E retry to finish (no restart/kill), then retry 16D downstream.
set -euo pipefail

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

POLL_SEC="${POLL_SEC:-120}"
LOG="logs/round16_after_16e_16d_downstream.log"
ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
E_MANIFEST="${ROUND16_ROOT}/manifests/stage16e_finetune_dispatch_manifest.csv"

mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

e_status_line() {
  python3 - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/round16_bruteforce/manifests/stage16e_finetune_dispatch_manifest.csv")
vc = df["status"].value_counts().to_dict()
print(
    f"success={vc.get('success', 0)} "
    f"running={vc.get('running', 0)} "
    f"pending={vc.get('pending', 0)} "
    f"failed={vc.get('failed', 0)}"
)
PY
}

e_manifest_idle() {
  python3 - <<'PY'
import pandas as pd
import sys

df = pd.read_csv("result/optimization_runs/round16_bruteforce/manifests/stage16e_finetune_dispatch_manifest.csv")
pending = int((df["status"] == "pending").sum())
running = int((df["status"] == "running").sum())
sys.exit(0 if pending == 0 and running == 0 else 1)
PY
}

retry_16e_running() {
  pgrep -f "bash tools/run_round16_retry_16e_failed.sh" >/dev/null
}

log "========== WAIT 16E (no restart) -> RETRY 16D DOWNSTREAM =========="
log "Watcher started; will not stop or restart the current 16E retry."

while true; do
  status_line="$(e_status_line)"
  retry_active="no"
  if retry_16e_running; then
    retry_active="yes"
  fi

  if e_manifest_idle && ! retry_16e_running; then
    log "16E complete (${status_line}); retry script exited."
    break
  fi

  log "16E in progress (${status_line}, retry_script=${retry_active}); sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

log "Starting 16D downstream retry (16E left untouched)."
bash tools/run_round16_retry_16d_downstream.sh 2>&1 | tee -a logs/round16_16d_downstream_after_16e.log

r16_notify --event pipeline-done --stages "16e,16d"
log "========== 16E -> 16D AUTO CONTINUATION DONE =========="
