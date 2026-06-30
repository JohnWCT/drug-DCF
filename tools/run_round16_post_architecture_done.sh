#!/usr/bin/env bash
# After 16F/16E finish on the legacy orchestrator: stop parent, run 16D, notify done.
set -euo pipefail

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-16}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-8}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

if pgrep -f "bash tools/run_round16_pipeline.sh" >/dev/null; then
  pkill -f "bash tools/run_round16_pipeline.sh" || true
  sleep 2
  echo "Stopped legacy run_round16_pipeline.sh orchestrator."
fi

bash tools/run_round16_pretrain_vicreg_stage16d.sh

python3 tools/round16_telegram_notify.py \
  --event pipeline-done \
  --stages "16f,16e,16d" || true

echo "Architecture round (16F/16E/16D) complete. Run tools/run_round16_downstream_sweep.sh in a future round for 16A–16C."
