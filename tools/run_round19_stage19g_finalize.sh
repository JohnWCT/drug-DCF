#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "${ROOT}"
: "${ROUND19G_EXPERIMENT_LOCK:?Set immutable 19G experiment lock path}"
python3 tools/round19_stage19g_executor.py \
  --experiment-lock "${ROUND19G_EXPERIMENT_LOCK}" \
  --output-root "${ROUND19G_OUTPUT_DIR:-result/optimization_runs/round19_factorial/stage19g}" \
  --finalize
