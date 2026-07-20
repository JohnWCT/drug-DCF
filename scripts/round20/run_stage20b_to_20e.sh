#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
R="result/optimization_runs/round20_unseen_drug_closure"
notify() { python3 tools/telegram_notify.py --message "$1" || true; }
DECISION="${R}/stage20a_dimension/stage20a_dimension_decision.json"

echo "Waiting for Stage 20A decision..."
while [[ ! -f "${DECISION}" ]]; do sleep 60; done
notify "[Round20] Stage 20A LOCKED — starting 20B→20E"

python3 tools/round20_stage20b.py build-manifest --dimension-lock "${DECISION}"
python3 tools/round20_dispatch.py --manifest "${R}/stage20b_predictor/manifest.jsonl" --max-parallel 16 --stage-label 20B
python3 tools/round20_stage20b.py analyze --stage-dir "${R}/stage20b_predictor"
notify "[Round20] Stage 20B done — locking model"

python3 tools/round20_model_lock.py \
  --stage20a-decision "${DECISION}" \
  --stage20b-guardrails "${R}/stage20b_predictor/stage20b_guardrail_report.json" \
  --output "${R}/stage20c_lock/final_model_lock.json" --strict
notify "[Round20] Stage 20C LOCKED — TCGA inference"

python3 scripts/round20/run_round20_tcga.py \
  --model-lock "${R}/stage20c_lock/final_model_lock.json" \
  --output-dir "${R}/stage20d_tcga" --strict
notify "[Round20] Stage 20D done — release"

python3 scripts/round20/build_round20_release.py \
  --model-lock "${R}/stage20c_lock/final_model_lock.json" \
  --tcga-dir "${R}/stage20d_tcga" \
  --output-dir "${R}/stage20e_release"
python3 scripts/round20/audit_round20_release.py --release-dir "${R}/stage20e_release" --strict
notify "[Round20] ALL DONE — unseen-drug closure complete"
