#!/usr/bin/env bash
# Stage 19B formal screening — parallel GPU packing to saturate a single large GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
MANIFEST="${OUTDIR}/manifests/stage19b_drug_predictor_manifest.csv"
STATUS_CSV="${OUTDIR}/manifests/stage19b_job_status.csv"
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

# Pack aggressively on large GPUs (override with ROUND19_JOBS_PER_GPU).
JOBS_PER_GPU="${ROUND19_JOBS_PER_GPU:-}"
if [[ -z "${JOBS_PER_GPU}" ]]; then
  JOBS_PER_GPU="$(python3 - <<'PY'
from tools.round19_oom_runner import recommend_jobs_per_gpu
print(recommend_jobs_per_gpu(target_util_frac=0.90, reserve_mb=3072, est_job_mb=2800))
PY
)"
fi

# Under packing, start mid-large then fall back on OOM exit 42.
MICRO_CANDS="${ROUND19_MICRO_BATCH_CANDIDATES:-256,128,64,32}"

echo "========== ROUND19 STAGE 19B PARALLEL DISPATCH $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "jobs_per_gpu=${JOBS_PER_GPU} micro=${MICRO_CANDS}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true

python3 tools/round19_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 19b_manifest
python3 - <<PY
import pandas as pd
df = pd.read_csv("${MANIFEST}")
assert len(df) == 117, len(df)
assert set(df.omics_id) == {"O1", "O2", "O3"}
print("manifest 117 OK")
PY

exec python3 tools/round19_oom_runner.py dispatch \
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
