#!/usr/bin/env bash
# Round 16 downstream brute-force only: 16A -> 16B -> 16C (run after architecture stages).
set -euo pipefail

export ROUND16_SKIP_DOWNSTREAM=0
rm -f config/round16_defer_downstream.flag

export ROUND16_PIPELINE_STAGES="${ROUND16_PIPELINE_STAGES:-16a,16b,16c}"
export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-16}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

exec bash tools/run_round16_pipeline.sh
