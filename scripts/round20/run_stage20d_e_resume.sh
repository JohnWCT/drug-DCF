#!/usr/bin/env bash
# Resume Round 20 from Stage 20D (after 20C lock already exists).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
R="result/optimization_runs/round20_unseen_drug_closure"
LOG_DIR="${R}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/stage20d_e_resume_$(date -u +%Y%m%dT%H%M%SZ).log"
notify() { python3 tools/telegram_notify.py --message "$1" || true; }

{
  echo "========== RESUME 20D-20E $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  notify "[Round20] Stage 20D TCGA inference starting (C32 + B_E3 ensemble)"
  python3 scripts/round20/run_round20_tcga.py \
    --model-lock "${R}/stage20c_lock/final_model_lock.json" \
    --output-dir "${R}/stage20d_tcga" --strict
  notify "[Round20] Stage 20D done — building release"
  python3 scripts/round20/build_round20_release.py \
    --model-lock "${R}/stage20c_lock/final_model_lock.json" \
    --tcga-dir "${R}/stage20d_tcga" \
    --output-dir "${R}/stage20e_release"
  python3 scripts/round20/audit_round20_release.py \
    --release-dir "${R}/stage20e_release" --strict
  notify "[Round20] ALL DONE — unseen-drug closure complete (C32 + pooled E3)"
  echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "${LOG}"
