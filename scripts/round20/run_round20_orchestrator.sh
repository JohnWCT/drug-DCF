#!/usr/bin/env bash
# Chain Stage 20A → 20B → 20C → 20D → 20E with Telegram. Resume-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

R="result/optimization_runs/round20_unseen_drug_closure"
LOG_DIR="${R}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/orchestrator_$(date -u +%Y%m%dT%H%M%SZ).log"
notify() { python3 tools/telegram_notify.py --message "$1" || true; }

exec > >(tee -a "${LOG}") 2>&1
echo "========== ROUND20 ORCHESTRATOR $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

# --- 20A ---
if [[ ! -f "${R}/stage20a_dimension/stage20a_dimension_decision.json" ]]; then
  notify "[Round20] Orchestrator: Stage 20A dispatch"
  bash scripts/round20/run_stage20a_full.sh
else
  echo "Stage 20A already LOCKED — skip"
fi

# --- 20B ---
if [[ ! -f "${R}/stage20b_predictor/stage20b_guardrail_report.json" ]]; then
  notify "[Round20] Orchestrator: Stage 20B build+dispatch"
  python3 tools/round20_stage20b.py build-manifest \
    --dimension-lock "${R}/stage20a_dimension/stage20a_dimension_decision.json"
  python3 tools/round20_dispatch.py \
    --manifest "${R}/stage20b_predictor/manifest.jsonl" \
    --max-parallel 16 --stage-label 20B
  python3 tools/round20_stage20b.py analyze --stage-dir "${R}/stage20b_predictor"
  notify "[Round20] Stage 20B guardrails written"
else
  echo "Stage 20B already complete — skip"
fi

# --- 20C ---
if [[ ! -f "${R}/stage20c_lock/final_model_lock.json" ]]; then
  notify "[Round20] Orchestrator: Stage 20C model lock"
  python3 tools/round20_model_lock.py \
    --stage20a-decision "${R}/stage20a_dimension/stage20a_dimension_decision.json" \
    --stage20b-guardrails "${R}/stage20b_predictor/stage20b_guardrail_report.json" \
    --output "${R}/stage20c_lock/final_model_lock.json" --strict
else
  echo "Stage 20C already LOCKED — skip"
fi

# --- 20D ---
if [[ ! -f "${R}/stage20d_tcga/stage20d_tcga_summary.json" ]]; then
  notify "[Round20] Orchestrator: Stage 20D TCGA inference"
  python3 scripts/round20/run_round20_tcga.py \
    --model-lock "${R}/stage20c_lock/final_model_lock.json" \
    --output-dir "${R}/stage20d_tcga" --strict
else
  echo "Stage 20D already complete — skip"
fi

# --- 20E ---
if [[ ! -f "${R}/stage20e_release/RELEASE_MANIFEST.json" ]]; then
  notify "[Round20] Orchestrator: Stage 20E release"
  python3 scripts/round20/build_round20_release.py \
    --model-lock "${R}/stage20c_lock/final_model_lock.json" \
    --tcga-dir "${R}/stage20d_tcga" \
    --output-dir "${R}/stage20e_release"
  python3 scripts/round20/audit_round20_release.py \
    --release-dir "${R}/stage20e_release" --strict
fi

notify "[Round20] Orchestrator FINISHED — scenario-focused unseen-drug closure complete"
echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
