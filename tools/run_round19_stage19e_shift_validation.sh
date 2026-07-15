#!/usr/bin/env bash
# Stage 19E formal shift validation (one strategy at a time; ~90% GPU pack + Telegram).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
CAND_LOCK="${OUTDIR}/reports/round19_stage19e_candidate_lock.json"
EXP_LOCK="${OUTDIR}/reports/round19_stage19e_experiment_lock.json"
mkdir -p logs

r19_notify() { python3 tools/round19_telegram_notify.py "$@" || true; }

STRATEGY=""
SMOKE_ONLY=0
PILOT_ONLY=0
SETUP_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --strategy=*) STRATEGY="${arg#*=}" ;;
    --strategy) shift_next=1 ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    --pilot-only) PILOT_ONLY=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
  esac
  if [[ "${shift_next:-0}" -eq 1 && "${arg}" != "--strategy" ]]; then
    STRATEGY="${arg}"
    shift_next=0
  fi
done

# Support: --strategy cancer_type_heldout
prev=""
for arg in "$@"; do
  if [[ "${prev}" == "--strategy" ]]; then STRATEGY="${arg}"; fi
  prev="${arg}"
done

if [[ -z "${STRATEGY}" && "${SETUP_ONLY}" -eq 0 && "${SMOKE_ONLY}" -eq 0 ]]; then
  echo "Usage: $0 --strategy {cancer_type_heldout|drug_heldout|scaffold_heldout}" >&2
  exit 2
fi

echo "========== ROUND19 STAGE 19E ${STRATEGY:-setup} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/write_round19_stage19d_baseline.py --root "${OUTDIR}" >/dev/null
if [[ ! -f "${CAND_LOCK}" ]]; then
  python3 tools/round19_stage19e_selector.py --root "${OUTDIR}" --output "${CAND_LOCK}"
fi
if [[ ! -f "${EXP_LOCK}" ]]; then
  python3 tools/round19_config_builder.py \
    --settings "${SETTINGS}" \
    --outdir "${OUTDIR}" \
    --stage 19e \
    --candidate-lock "${CAND_LOCK}" \
    --write-experiment-lock
fi

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-done --stage 19e-setup --extra "experiment lock ready"
  exit 0
fi

MANIFEST="${OUTDIR}/manifests/stage19e_${STRATEGY}_manifest.csv"
STATUS_CSV="${OUTDIR}/manifests/stage19e_${STRATEGY}_job_status.csv"
test -f "${MANIFEST}"
test -f "${EXP_LOCK}"

N_JOBS="$(python3 - <<PY
import pandas as pd
print(len(pd.read_csv("${MANIFEST}")))
PY
)"

if [[ "${SMOKE_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-start --stage "19e-${STRATEGY}-smoke" --extra "${N_JOBS} jobs in manifest"
  python3 - <<PY
import pandas as pd, subprocess, sys
from pathlib import Path
root = Path("result/optimization_runs/round19_factorial")
df = pd.read_csv(root / "manifests" / "stage19e_${STRATEGY}_manifest.csv")
# one job per candidate fold0
for eid in sorted(df.candidate_id.unique()):
    r = df[(df.candidate_id == eid) & (df.fold_id == 0)].iloc[0]
    rd = root / "stage19e_smoke" / "${STRATEGY}" / str(r.job_id)
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
print("19E strategy smoke OK")
PY
  r19_notify --event stage-done --stage "19e-${STRATEGY}-smoke"
  exit 0
fi

if [[ "${PILOT_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-start --stage "19e-${STRATEGY}-pilot"
  python3 - <<PY
import pandas as pd, subprocess, sys
from pathlib import Path
root = Path("result/optimization_runs/round19_factorial")
df = pd.read_csv(root / "manifests" / "stage19e_${STRATEGY}_manifest.csv")
# primary atom + pooled comparator
for eid in ("E1", "E3"):
    row = df[(df.candidate_id == eid) & (df.fold_id == 0)]
    if row.empty:
        continue
    r = row.iloc[0]
    rd = root / "stage19e_pilot" / "${STRATEGY}" / str(r.job_id)
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
        "--max-epochs", "80",
        "--micro-batch-size", "256",
        "--accumulation-steps", "4",
    ]
    print("PILOT", r.job_id)
    subprocess.check_call(cmd)
print("19E pilot OK")
PY
  r19_notify --event stage-done --stage "19e-${STRATEGY}-pilot"
  exit 0
fi

# Formal: pack ~90% GPU
JOBS_PER_GPU="${ROUND19_JOBS_PER_GPU:-}"
if [[ -z "${JOBS_PER_GPU}" ]]; then
  JOBS_PER_GPU="$(python3 - <<'PY'
from tools.round19_oom_runner import recommend_jobs_per_gpu
print(recommend_jobs_per_gpu(target_util_frac=0.90, reserve_mb=3072, est_job_mb=2800))
PY
)"
fi
MICRO_CANDS="${ROUND19_MICRO_BATCH_CANDIDATES:-256,128,64,32}"
echo "formal strategy=${STRATEGY} n_jobs=${N_JOBS} pack=${JOBS_PER_GPU} micro=${MICRO_CANDS}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true

LOG="logs/round19_stage19e_${STRATEGY}_$(date -u +%Y%m%dT%H%M%SZ).log"
r19_notify --event stage-start --stage "19e-${STRATEGY}" \
  --extra "formal ${N_JOBS} jobs, pack=${JOBS_PER_GPU}/GPU ~90%" \
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
  --split-assignment "${OUTDIR}/splits/round19e_cancer_type_heldout_5cv.csv" \
  --internal-test-path "${OUTDIR}/splits/internal_test_split.csv" \
  2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e

if [[ "${RC}" -eq 0 ]]; then
  r19_notify --event stage-done --stage "19e-${STRATEGY}" --manifest "${MANIFEST}"
else
  r19_notify --event stage-fail --stage "19e-${STRATEGY}" --reason "dispatch_exit_${RC}" --manifest "${MANIFEST}"
fi
exit "${RC}"
