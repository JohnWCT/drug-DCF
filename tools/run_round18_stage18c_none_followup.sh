#!/usr/bin/env bash
# Round 18 Stage 18C-B: best pure + best residual × none × 3 folds (=6 jobs)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=tools/run_round18_notify_helpers.sh
source "$(dirname "$0")/run_round18_notify_helpers.sh"

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-6}"
ROUND18_NUM_WORKERS="${ROUND18_NUM_WORKERS:-0}"
export ROUND18_NUM_WORKERS
export ROUND18_PIN_MEMORY="${ROUND18_PIN_MEMORY:-1}"
CURRENT_STAGE="18C-B"

trap 'r18_notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code $?"' ERR

echo "========== ROUND18 STAGE 18C-B NONE FOLLOWUP START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU} ROUND18_NUM_WORKERS=${ROUND18_NUM_WORKERS}"

# Refresh ranking + write top_for_none.json from completed 18C-A
python tools/analyze_round18.py --outdir "${OUTDIR}" --settings "${SETTINGS}"

TOP_JSON="${OUTDIR}/reports/round18_18c_top_for_none.json"
if [[ ! -f "${TOP_JSON}" ]]; then
  echo "Missing ${TOP_JSON}; cannot build none follow-up" >&2
  exit 1
fi

python tools/round18_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 18c_none

MANIFEST="${OUTDIR}/manifests/stage18c_none_followup_manifest.csv"
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

top = json.loads(Path("result/optimization_runs/round18_architecture/reports/round18_18c_top_for_none.json").read_text())
cands = top["top_cross_attention_for_none"]
assert len(cands) == 2, cands
modes = {c["residual_mode"] for c in cands}
assert modes == {"pure", "pooled_residual"}, modes
df = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18c_none_followup_manifest.csv")
assert len(df) == 6, len(df)
assert set(df["omics_mode"]) == {"none"}
assert set(df["residual_mode"]) == {"pure", "pooled_residual"}
print("18C-B manifest OK: 6 jobs")
print(df[["job_id", "architecture_id", "residual_mode", "fold_id"]].to_string(index=False))
PY

r18_notify --event stage-start --stage "${CURRENT_STAGE}" \
  --extra "6 jobs (best pure + best residual × none × 3fold); MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU}"

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18c_none_followup_manifest.csv")
row = df.iloc[0].to_dict()
row["job_id"] = str(row["job_id"]) + "_smoke"
row["result_dir"] = (
    "result/optimization_runs/round18_architecture/stage18c_none_smoke/"
    f"{row['architecture_id']}/fold_0"
)
row["requested_micro_batch"] = 16
row["max_epochs"] = 1
row["max_batches"] = 2
out = Path("result/optimization_runs/round18_architecture/manifests/stage18c_none_smoke_onejob.csv")
pd.DataFrame([row]).to_csv(out, index=False)
print("wrote", out)
PY
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18c_none_smoke_onejob.csv" \
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

r18_notify --event stage-done --stage "${CURRENT_STAGE}" --manifest "${MANIFEST}"
echo "========== ROUND18 STAGE 18C-B DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
