#!/usr/bin/env bash
# Round 18 Stage 18D: formal 5CV for locked 5 candidates (=25 jobs)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=tools/run_round18_notify_helpers.sh
source "$(dirname "$0")/run_round18_notify_helpers.sh"

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
WRITE_LOCK="${WRITE_LOCK:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-8}"
ROUND18_NUM_WORKERS="${ROUND18_NUM_WORKERS:-0}"
export ROUND18_NUM_WORKERS
export ROUND18_PIN_MEMORY="${ROUND18_PIN_MEMORY:-1}"
CURRENT_STAGE="18D"
LOCK="${OUTDIR}/reports/round18_locked_selection.json"

trap 'r18_notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code $?"' ERR

echo "========== ROUND18 STAGE 18D FORMAL 5CV START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

if [[ "${WRITE_LOCK}" == "1" ]]; then
  echo "[18D] writing formal lock (requires 18B+18C-A+18C-B complete)"
  python tools/analyze_round18.py --outdir "${OUTDIR}" --settings "${SETTINGS}" --write-lock
  r18_notify --event lock-written --lock "${LOCK}"
fi

if [[ ! -f "${LOCK}" ]]; then
  echo "Missing ${LOCK}; refuse placeholder 18D. Set WRITE_LOCK=1 after 18C-B." >&2
  exit 1
fi

python tools/round18_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 18d

MANIFEST="${OUTDIR}/manifests/stage18d_formal_5cv_manifest.csv"
python - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18d_formal_5cv_manifest.csv")
assert len(df) == 25, len(df)
assert df["fold_id"].nunique() == 5
assert df["architecture_id"].nunique() == 5
print("18D manifest OK: 25 jobs")
print(df.groupby("architecture_id").size().to_string())
PY

r18_notify --event stage-start --stage "${CURRENT_STAGE}" \
  --extra "25 jobs (5 candidates × 5fold); MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU}"

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18d_formal_5cv_manifest.csv")
row = df.iloc[0].to_dict()
row["job_id"] = str(row["job_id"]) + "_smoke"
row["result_dir"] = (
    "result/optimization_runs/round18_architecture/stage18d_smoke/"
    f"{row['architecture_id']}/fold_0"
)
row["requested_micro_batch"] = 16
row["max_epochs"] = 1
row["max_batches"] = 2
out = Path("result/optimization_runs/round18_architecture/manifests/stage18d_smoke_onejob.csv")
pd.DataFrame([row]).to_csv(out, index=False)
print("wrote", out)
PY
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18d_smoke_onejob.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 \
    --micro-batch-candidates 16,8 \
    --limit 1
else
  python tools/round18_oom_runner.py dispatch \
    --manifest "${MANIFEST}" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" \
    --micro-batch-candidates 512,256,128,64,32
fi

python tools/analyze_round18.py --outdir "${OUTDIR}" --settings "${SETTINGS}"
r18_notify --event stage-done --stage "${CURRENT_STAGE}" --manifest "${MANIFEST}"
echo "========== ROUND18 STAGE 18D DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
