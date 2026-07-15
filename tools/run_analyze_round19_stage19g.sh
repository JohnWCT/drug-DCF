#!/usr/bin/env bash
# Formal analysis wrapper; all paths are explicit and completeness is mandatory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

: "${ROUND19G_OUTPUT_DIR:?required}"
: "${ROUND19G_CASE_MANIFEST:?required}"
: "${ROUND19G_FINAL_LOCK:?required}"
: "${ROUND19G_EXPERIMENT_LOCK:?required}"
: "${ROUND19G_VERDICT:?required: SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED}"

python3 tools/analyze_round19_stage19g.py \
  --output-dir "${ROUND19G_OUTPUT_DIR}" \
  --case-manifest "${ROUND19G_CASE_MANIFEST}" \
  --final-lock "${ROUND19G_FINAL_LOCK}" \
  --experiment-lock "${ROUND19G_EXPERIMENT_LOCK}" \
  --verdict "${ROUND19G_VERDICT}" \
  --require-complete
