#!/usr/bin/env bash
# Stage 19D — repeated ModelID-grouped 5CV confirmation (seeds 52/62/72).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
PROPOSAL="${OUTDIR}/reports/round19_stage19d_candidate_proposal.json"
MANIFEST="${OUTDIR}/manifests/stage19d_manifest.csv"
STATUS_CSV="${OUTDIR}/manifests/stage19d_job_status.csv"
LOCK="${OUTDIR}/reports/round19_stage19d_experiment_lock.json"
mkdir -p logs

r19_notify() { python3 tools/round19_telegram_notify.py "$@" || true; }

SMOKE_ONLY=0
PILOT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --pilot-only) PILOT_ONLY=1 ;;
  esac
done

echo "========== ROUND19 STAGE 19D $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

# Baseline + proposal
python3 tools/write_round19_stage19c_baseline.py --root "${OUTDIR}" >/dev/null
if [[ ! -f "${PROPOSAL}" ]]; then
  python3 tools/round19_stage19d_selector.py --root "${OUTDIR}" --output "${PROPOSAL}"
fi

python3 tools/round19_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 19d \
  --candidate-proposal "${PROPOSAL}" \
  --split-seeds 52,62,72 \
  --n-folds 5 \
  --write-experiment-lock

N_JOBS="$(python3 - <<PY
import pandas as pd
print(len(pd.read_csv("${MANIFEST}")))
PY
)"
N_CAND="$(python3 - <<PY
import json
from pathlib import Path
print(json.loads(Path("${PROPOSAL}").read_text())["n_candidates"])
PY
)"

if [[ "${SMOKE_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-start --stage 19d-smoke --extra "${N_CAND} cand / ${N_JOBS} jobs planned"
  python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd
import subprocess
import sys

root = Path("result/optimization_runs/round19_factorial")
df = pd.read_csv(root / "manifests" / "stage19d_manifest.csv")
# one job per mandatory-ish candidate covering seeds/folds diversity
wanted = [
    ("F0_historical_anchor", 52, 0),
    ("F1_primary_o2", 62, 1),
    ("F2_full_omics_o3", 72, 2),
    ("F3_best_pooled_o2", 52, 3),
    ("F4_source_only_o4", 62, 4),
]
for cid, seed, fold in wanted:
    row = df[(df.candidate_id == cid) & (df.split_seed == seed) & (df.fold_id == fold)]
    if row.empty:
        raise SystemExit(f"missing smoke row {cid} seed{seed} fold{fold}")
    r = row.iloc[0]
    rd = root / "stage19d_smoke" / str(r.job_id)
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
print("19D smoke OK")
PY
  r19_notify --event stage-done --stage 19d-smoke
  exit 0
fi

if [[ "${PILOT_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-start --stage 19d-pilot --extra "1 fold per mandatory candidate"
  python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd
import subprocess
import sys

root = Path("result/optimization_runs/round19_factorial")
df = pd.read_csv(root / "manifests" / "stage19d_manifest.csv")
prop = json.loads((root / "reports" / "round19_stage19d_candidate_proposal.json").read_text())
mandatory = [c["candidate_id"] for c in prop["candidates"] if c.get("mandatory")]
for i, cid in enumerate(mandatory):
    row = df[(df.candidate_id == cid) & (df.split_seed == 52) & (df.fold_id == 0)]
    if row.empty:
        raise SystemExit(f"missing pilot {cid}")
    r = row.iloc[0]
    rd = root / "stage19d_pilot" / str(r.job_id)
    rd.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "step1_finetune_latent_pipeline_round19.py",
        "--mode", "train_fold", "--pilot",
        "--settings", "config/round19_factorial_settings.json",
        "--result-dir", str(rd),
        "--response-path", str(root / "data" / "round19_eligible_response.csv"),
        "--split-assignment", str(r.split_assignment_path),
        "--internal-test-path", str(root / "splits" / "internal_test_split.csv"),
        "--drug-id", str(r.drug_id),
        "--predictor-id", str(r.predictor_id),
        "--omics-id", str(r.omics_id),
        "--fold-id", str(int(r.fold_id)),
        "--model-seed", "101",
        "--max-epochs", "80",
        "--early-stop-patience", "20",
        "--early-stop-start-epoch", "10",
        "--micro-batch-size", "256",
        "--accumulation-steps", "4",
    ]
    print("PILOT", r.job_id)
    subprocess.check_call(cmd)
print("19D pilot OK")
PY
  r19_notify --event stage-done --stage 19d-pilot
  exit 0
fi

# Formal: pack GPU ~90% (override with ROUND19_JOBS_PER_GPU)
JOBS_PER_GPU="${ROUND19_JOBS_PER_GPU:-}"
if [[ -z "${JOBS_PER_GPU}" ]]; then
  JOBS_PER_GPU="$(python3 - <<'PY'
from tools.round19_oom_runner import recommend_jobs_per_gpu
print(recommend_jobs_per_gpu(target_util_frac=0.90, reserve_mb=3072, est_job_mb=2800))
PY
)"
fi
MICRO_CANDS="${ROUND19_MICRO_BATCH_CANDIDATES:-256,128,64,32}"
echo "formal n_jobs=${N_JOBS} n_cand=${N_CAND} pack=${JOBS_PER_GPU} micro=${MICRO_CANDS}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
test -f "${LOCK}"
r19_notify --event stage-start --stage 19d \
  --extra "formal ${N_JOBS} jobs (${N_CAND} cand × 15), pack=${JOBS_PER_GPU}/GPU ~90%" \
  --manifest "${MANIFEST}"

set +e
python3 tools/round19_oom_runner.py dispatch \
  --manifest "${MANIFEST}" \
  --pipeline step1_finetune_latent_pipeline_round19.py \
  --max-jobs-per-gpu "${JOBS_PER_GPU}" \
  --no-auto-pack \
  --micro-batch-candidates "${MICRO_CANDS}" \
  --target-effective-batch 1024 \
  --status-csv "${STATUS_CSV}" \
  --settings "${SETTINGS}" \
  --response-path "${OUTDIR}/data/round19_eligible_response.csv" \
  --split-assignment "${OUTDIR}/splits/round19d_seed52_5cv_assignments.csv" \
  --internal-test-path "${OUTDIR}/splits/internal_test_split.csv"
RC=$?
set -e
if [[ "${RC}" -eq 0 ]]; then
  r19_notify --event stage-done --stage 19d --manifest "${MANIFEST}"
else
  r19_notify --event stage-fail --stage 19d --reason "dispatch_exit_${RC}" --manifest "${MANIFEST}"
fi
exit "${RC}"
