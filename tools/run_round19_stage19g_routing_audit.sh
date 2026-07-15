#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "${ROOT}"
: "${ROUND19G_EXPERIMENT_LOCK:?Set immutable 19G experiment lock path}"
MODE="${ROUND19G_MODE:-pilot}"
EXECUTE=(); [[ "${ROUND19G_EXECUTE:-0}" == "1" ]] && EXECUTE=(--execute)
python3 tools/round19_stage19g_dispatch.py "--${MODE}" \
  --experiment-lock "${ROUND19G_EXPERIMENT_LOCK}" --methods routing "${EXECUTE[@]}"
