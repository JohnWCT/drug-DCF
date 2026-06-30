#!/usr/bin/env bash
# Continue 16E -> 16D after 16F finetune is already complete.
set -euo pipefail

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"
export PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-12}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

LOG="logs/round16_ed_continuation.log"
mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

notify() {
  python3 tools/round16_telegram_notify.py "$@" || true
}

log "========== CONTINUE 16E -> 16D =========="

if pgrep -f "bash tools/run_round16_pipeline.sh" >/dev/null; then
  log "Stopping legacy pipeline orchestrator"
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
log "========== 16E -> 16D DONE =========="
