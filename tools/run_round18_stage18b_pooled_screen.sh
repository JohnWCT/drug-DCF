#!/usr/bin/env bash
# Round 18 Stage 18B: build pooled screening manifest + dispatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-1}"
LIMIT_JOBS="${LIMIT_JOBS:-}"
JOB_FILTER="${JOB_FILTER:-}"

echo "========== ROUND18 STAGE 18B POOLED SCREEN START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "SMOKE_ONLY=${SMOKE_ONLY}"

if [[ ! -f "${OUTDIR}/data/round18_eligible_response.csv" ]]; then
  python tools/round18_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 18a
fi

echo "[18B] feature ModelID coverage preflight"
python tools/round18_feature_coverage.py --settings "${SETTINGS}" --outdir "${OUTDIR}"

python tools/round18_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 18b

python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("result/optimization_runs/round18_architecture/manifests/stage18b_screening_manifest.csv")
df = pd.read_csv(p)
assert len(df) == 45, len(df)
families = set(df["architecture_family"])
assert "pooled_mlp" in families and "pooled_transformer" in families
tf_ids = set(df["transformer_config_id"].dropna().astype(str))
assert "P0_historical_hparams_corrected_mask" in tf_ids
print(f"18B manifest OK: {len(df)} jobs, families={sorted(families)}")
print(df.groupby(["architecture_family", "omics_mode"]).size().to_string())
PY

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  echo "[18B] SMOKE_ONLY=1: synthetic smoke + one short real train_fold via OOM runner"
  python step1_finetune_latent_pipeline_round18_cv.py --mode smoke --outdir "${OUTDIR}" --steps 2
  python - <<'PY'
import pandas as pd
from pathlib import Path

src = Path("result/optimization_runs/round18_architecture/manifests/stage18b_screening_manifest.csv")
df = pd.read_csv(src)
row = df[
    (df.architecture_family == "pooled_mlp")
    & (df.omics_mode == "own_plus_summary")
    & (df.fold_id == 0)
].iloc[0].to_dict()
row["result_dir"] = "result/optimization_runs/round18_architecture/stage18b_smoke/pooled_mlp_own_plus_summary_f0"
row["requested_micro_batch"] = 16
row["max_epochs"] = 1
row["max_batches"] = 2
out = Path("result/optimization_runs/round18_architecture/manifests/stage18b_smoke_onejob.csv")
pd.DataFrame([row]).to_csv(out, index=False)
print("wrote", out)
PY
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18b_smoke_onejob.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 \
    --micro-batch-candidates 16,8 \
    --limit 1
else
  echo "[18B] SMOKE_ONLY=0: dispatching full screening manifest"
  EXTRA=()
  if [[ -n "${LIMIT_JOBS}" ]]; then EXTRA+=(--limit "${LIMIT_JOBS}"); fi
  if [[ -n "${JOB_FILTER}" ]]; then EXTRA+=(--job-filter "${JOB_FILTER}"); fi
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18b_screening_manifest.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 \
    --micro-batch-candidates 512,256,128,64,32 \
    "${EXTRA[@]}"
fi

echo "========== ROUND18 STAGE 18B POOLED SCREEN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
