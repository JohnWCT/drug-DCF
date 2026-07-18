#!/usr/bin/env bash
# Stage 20A full execution: 30 paired C16/C32 jobs + analysis + Telegram.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

STAGE_DIR="result/optimization_runs/round20_unseen_drug_closure/stage20a_dimension"
LOG_DIR="result/optimization_runs/round20_unseen_drug_closure/logs"
mkdir -p "${LOG_DIR}" "${STAGE_DIR}"
LOG="${LOG_DIR}/stage20a_full_$(date -u +%Y%m%dT%H%M%SZ).log"

notify() { python3 tools/telegram_notify.py --message "$1" || true; }

{
  echo "========== STAGE 20A FULL $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  python3 scripts/round20/build_stage20a_manifest.py --dry-run
  notify "[Round20 Stage 20A] launching 30-job dispatch (max_parallel=16, resume=true)"
  python3 tools/round20_dispatch.py \
    --manifest "${STAGE_DIR}/manifest.jsonl" \
    --max-parallel 16 \
    --stage-label 20A \
    --micro-batch-size 256 \
    --accumulation-steps 4
  echo "========== ANALYZE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  python3 scripts/round20/analyze_stage20a.py --input-dir "${STAGE_DIR}" --strict
  notify "[Round20 Stage 20A] analysis LOCKED — see ${STAGE_DIR}/stage20a_dimension_decision.json"
} 2>&1 | tee -a "${LOG}"

echo "LOG=${LOG}"
