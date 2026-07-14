#!/usr/bin/env bash
# Stage 19C — O0/O4 completion + shuffled-context controls.
# Formal mode packs GPU ~90% (override with ROUND19_JOBS_PER_GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
LOCK="${OUTDIR}/reports/round19_stage19c_candidate_lock.json"
MANIFEST="${OUTDIR}/manifests/stage19c_manifest.csv"
STATUS_CSV="${OUTDIR}/manifests/stage19c_job_status.csv"
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

r19_notify() {
  python3 tools/round19_telegram_notify.py "$@" || true
}

SMOKE_ONLY=0
PILOT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --pilot-only) PILOT_ONLY=1 ;;
  esac
done

echo "========== ROUND19 STAGE 19C $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "smoke=${SMOKE_ONLY} pilot=${PILOT_ONLY}"

if [[ ! -f "${LOCK}" ]]; then
  python3 tools/round19_stage19c_selector.py \
    --root "${OUTDIR}" \
    --require-complete \
    --expected-jobs 117 \
    --output "${LOCK}" \
    --write-baseline
fi

python3 tools/round19_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 19c \
  --candidate-lock "${LOCK}" \
  --include-context-controls

if [[ "${SMOKE_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-start --stage 19c-smoke --extra "data_smoke O0/O4/O2_shuffle"
  python3 step1_finetune_latent_pipeline_round19.py \
    --mode data_smoke \
    --settings "${SETTINGS}" \
    --result-dir "${OUTDIR}/stage19c_smoke/D0__P2__O0__fold0" \
    --drug-id D0 --predictor-id P2 --omics-id O0 --fold-id 0 \
    --max-batches 2 --max-rows 32

  python3 step1_finetune_latent_pipeline_round19.py \
    --mode data_smoke \
    --settings "${SETTINGS}" \
    --result-dir "${OUTDIR}/stage19c_smoke/D0__P0__O4__fold0" \
    --drug-id D0 --predictor-id P0 --omics-id O4 --fold-id 0 \
    --max-batches 2 --max-rows 32

  python3 step1_finetune_latent_pipeline_round19.py \
    --mode data_smoke \
    --settings "${SETTINGS}" \
    --result-dir "${OUTDIR}/stage19c_smoke/D0__P2__O2__ctx_shuffle__fold0" \
    --drug-id D0 --predictor-id P2 --omics-id O2 --fold-id 0 \
    --control-type context_shuffle \
    --train-shuffle-seed 19032 \
    --val-shuffle-seed 19033 \
    --max-batches 2 --max-rows 32

  r19_notify --event stage-done --stage 19c-smoke
  echo "19C smoke OK"
  exit 0
fi

if [[ "${PILOT_ONLY}" -eq 1 ]]; then
  r19_notify --event stage-start --stage 19c-pilot --extra "4 full train_fold jobs"
  POOLED="$(python3 - <<PY
import json
from pathlib import Path
lock = json.loads(Path("${LOCK}").read_text())
bp = lock["best_pooled_for_shuffle"]
print(bp["drug_id"], bp["predictor_id"])
PY
)"
  read -r POOLED_D POOLED_P <<< "${POOLED}"

  pilot_jobs=(
    "D0|P2|O0|0|none"
    "D0|P2|O4|0|none"
    "D0|P2|O2|0|context_shuffle|19032|19033"
    "${POOLED_D}|${POOLED_P}|O3|0|context_shuffle|19032|19033"
  )
  for spec in "${pilot_jobs[@]}"; do
    IFS='|' read -r D P O F CTRL TRS VRS <<< "${spec}"
    if [[ "${CTRL}" == "context_shuffle" ]]; then
      SUF="__ctx_shuffle"
    else
      SUF=""
    fi
    RD="${OUTDIR}/stage19c_pilot/${D}__${P}__${O}__fold${F}${SUF}"
    mkdir -p "${RD}"
    CMD=(python3 step1_finetune_latent_pipeline_round19.py --mode train_fold --pilot
      --settings "${SETTINGS}" --result-dir "${RD}"
      --response-path "${OUTDIR}/data/round19_eligible_response.csv"
      --split-assignment "${OUTDIR}/splits/screening_3fold_assignments.csv"
      --internal-test-path "${OUTDIR}/splits/internal_test_split.csv"
      --drug-id "${D}" --predictor-id "${P}" --omics-id "${O}" --fold-id "${F}"
      --max-epochs 80 --early-stop-patience 20 --early-stop-start-epoch 10
      --micro-batch-size 256 --accumulation-steps 4)
    if [[ "${CTRL}" == "context_shuffle" ]]; then
      CMD+=(--control-type context_shuffle --train-shuffle-seed "${TRS}" --val-shuffle-seed "${VRS}")
    fi
    echo "Pilot: ${RD}"
    "${CMD[@]}"
  done
  r19_notify --event stage-done --stage 19c-pilot
  echo "19C pilot OK"
  exit 0
fi

# Formal dispatch — pack to ~90% GPU utilization
JOBS_PER_GPU="${ROUND19_JOBS_PER_GPU:-}"
if [[ -z "${JOBS_PER_GPU}" ]]; then
  JOBS_PER_GPU="$(python3 - <<'PY'
from tools.round19_oom_runner import recommend_jobs_per_gpu
print(recommend_jobs_per_gpu(target_util_frac=0.90, reserve_mb=3072, est_job_mb=2800))
PY
)"
fi
MICRO_CANDS="${ROUND19_MICRO_BATCH_CANDIDATES:-256,128,64,32}"

N_JOBS="$(python3 - <<PY
import pandas as pd
print(len(pd.read_csv("${MANIFEST}")))
PY
)"
N_CELLS="$(python3 - <<PY
import json
from pathlib import Path
lock=json.loads(Path("${LOCK}").read_text())
print(len(lock.get("unique_cells") or lock.get("selected_cells") or []))
PY
)"

echo "jobs_per_gpu=${JOBS_PER_GPU} micro=${MICRO_CANDS} n_jobs=${N_JOBS} n_cells=${N_CELLS}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true

r19_notify --event stage-start --stage 19c \
  --extra "formal dispatch: ${N_JOBS} jobs, ${N_CELLS} cells, pack=${JOBS_PER_GPU}/GPU" \
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
  --split-assignment "${OUTDIR}/splits/screening_3fold_assignments.csv" \
  --internal-test-path "${OUTDIR}/splits/internal_test_split.csv"
RC=$?
set -e

if [[ "${RC}" -eq 0 ]]; then
  r19_notify --event stage-done --stage 19c --manifest "${MANIFEST}"
else
  r19_notify --event stage-fail --stage 19c --reason "dispatch_exit_${RC}" --manifest "${MANIFEST}"
fi
exit "${RC}"
