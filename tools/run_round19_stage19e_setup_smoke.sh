#!/usr/bin/env bash
# Stage 19E setup smoke: candidate lock, groups, splits, manifests, tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
CAND_LOCK="${OUTDIR}/reports/round19_stage19e_candidate_lock.json"
mkdir -p logs

r19_notify() { python3 tools/round19_telegram_notify.py "$@" || true; }

echo "========== ROUND19 STAGE 19E SETUP SMOKE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
r19_notify --event stage-start --stage 19e-setup-smoke

python3 tools/write_round19_stage19d_baseline.py --root "${OUTDIR}"

python3 tools/round19_stage19e_selector.py --root "${OUTDIR}" --output "${CAND_LOCK}"

python3 tools/round19_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 19e \
  --candidate-lock "${CAND_LOCK}" \
  --write-experiment-lock

echo "[19E] running unit/integration tests"
pytest tests/test_round19_stage19e_*.py -q --tb=line

echo "[19E] data smoke (5 representative jobs)"
python3 - <<'PY'
import subprocess
import sys
from pathlib import Path
import pandas as pd

root = Path("result/optimization_runs/round19_factorial")
wanted = [
    ("cancer_type_heldout", "E1", 0),
    ("drug_heldout", "E0", 0),
    ("scaffold_heldout", "E2", 1),
    ("cancer_type_heldout", "E4", 2),
    ("drug_heldout", "E3", 0),
]
for strategy, eid, fold in wanted:
    man = root / "manifests" / f"stage19e_{strategy}_manifest.csv"
    df = pd.read_csv(man)
    row = df[(df.candidate_id == eid) & (df.fold_id == fold)]
    if row.empty:
        raise SystemExit(f"missing smoke row {strategy} {eid} fold{fold}")
    r = row.iloc[0]
    rd = root / "stage19e_smoke" / strategy / str(r.job_id)
    rd.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "step1_finetune_latent_pipeline_round19.py",
        "--mode", "data_smoke",
        "--settings", "config/round19_factorial_settings.json",
        "--result-dir", str(rd),
        "--response-path", str(root / "data" / "round19_eligible_response.csv"),
        "--split-assignment", str(r.split_assignment_path),
        "--internal-test-path", str(root / "splits" / "internal_test_split.csv"),
        "--drug-id", str(r.drug_id),
        "--predictor-id", str(r.predictor_id),
        "--omics-id", str(r.omics_id),
        "--fold-id", str(int(r.fold_id)),
        "--max-batches", "2", "--max-rows", "64",
    ]
    print("SMOKE", r.job_id)
    subprocess.check_call(cmd)
print("19E setup+data smoke OK")
PY

r19_notify --event stage-done --stage 19e-setup-smoke --extra "locks+splits+manifests+tests+data_smoke"
echo "========== ROUND19 STAGE 19E SETUP SMOKE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
