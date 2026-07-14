#!/usr/bin/env bash
# Round 18 Stage 18C: cross-attention screening (18C-A grid)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=tools/run_round18_notify_helpers.sh
source "$(dirname "$0")/run_round18_notify_helpers.sh"
CURRENT_STAGE="18C-A"
trap 'r18_notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code $?"' ERR

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-1}"
LIMIT_JOBS="${LIMIT_JOBS:-}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
ROUND18_NUM_WORKERS="${ROUND18_NUM_WORKERS:-0}"
export ROUND18_NUM_WORKERS
export ROUND18_PIN_MEMORY="${ROUND18_PIN_MEMORY:-1}"

echo "========== ROUND18 STAGE 18C START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "SMOKE_ONLY=${SMOKE_ONLY} MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU} LIMIT_JOBS=${LIMIT_JOBS:-all}"

python tools/round18_feature_coverage.py --settings "${SETTINGS}" --outdir "${OUTDIR}"

python tools/round18_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 18c

python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("result/optimization_runs/round18_architecture/manifests/stage18c_cross_attention_manifest.csv")
df = pd.read_csv(p)
assert set(df["architecture_family"]) == {"cross_attention"}
assert len(df) == 48, len(df)
assert set(df["omics_mode"]) == {"own_plus_summary", "own_proto_context_projected_16"}
assert set(df["residual_mode"]) == {"pure", "pooled_residual"}
print(f"18C-A manifest OK: {len(df)} jobs")
print(df.groupby(["transformer_config_id", "residual_mode", "omics_mode"]).size().to_string())
PY

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  echo "[18C] SMOKE_ONLY=1: one short cross-attn job"
  python - <<'PY'
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "result/optimization_runs/round18_architecture/manifests/stage18c_cross_attention_manifest.csv"
)
pref = df[
    (df["residual_mode"] == "pure") & (df["omics_mode"] == "own_plus_summary")
]
row = (pref if len(pref) else df).iloc[0].to_dict()
row["job_id"] = str(row["job_id"]) + "_smoke"
row["result_dir"] = (
    "result/optimization_runs/"
    "round18_architecture/stage18c_smoke/"
    f"{row['architecture_id']}/fold_0"
)
row["requested_micro_batch"] = 16
row["max_epochs"] = 1
row["max_batches"] = 2
out = Path(
    "result/optimization_runs/round18_architecture/manifests/stage18c_smoke_onejob.csv"
)
pd.DataFrame([row]).to_csv(out, index=False)
print("wrote", out)
print("architecture_id=", row["architecture_id"])
print("result_dir=", row["result_dir"])
PY
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18c_smoke_onejob.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 \
    --micro-batch-candidates 16,8 \
    --limit 1
else
  echo "[18C] SMOKE_ONLY=0: dispatching 18C-A screening manifest"
  r18_notify --event stage-start --stage "${CURRENT_STAGE}" --extra "48-job cross-attention screen; MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU}"
  EXTRA=()
  if [[ -n "${LIMIT_JOBS}" ]]; then EXTRA+=(--limit "${LIMIT_JOBS}"); fi
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18c_cross_attention_manifest.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" \
    --micro-batch-candidates 512,256,128,64,32 \
    "${EXTRA[@]}"
fi

r18_notify --event stage-done --stage "${CURRENT_STAGE}" --manifest "${OUTDIR}/manifests/stage18c_cross_attention_manifest.csv"
echo "========== ROUND18 STAGE 18C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
