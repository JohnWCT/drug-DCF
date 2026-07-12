#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-1}"
LIMIT_JOBS="${LIMIT_JOBS:-}"
echo "========== ROUND18 STAGE 18C START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
python tools/round18_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 18c
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("result/optimization_runs/round18_architecture/manifests/stage18c_cross_attention_manifest.csv")
df = pd.read_csv(p)
assert set(df["architecture_family"]) == {"cross_attention"}
print(f"18C manifest OK: {len(df)} jobs")
PY
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18c_cross_attention_manifest.csv")
row = df.iloc[0].to_dict()
row["job_id"] = str(row["job_id"]) + "_smoke"
row["result_dir"] = f"result/optimization_runs/round18_architecture/stage18c_smoke/{row[architecture_id]}/fold_0"
row["requested_micro_batch"] = 16
row["max_epochs"] = 1
row["max_batches"] = 2
Path("result/optimization_runs/round18_architecture/manifests/stage18c_smoke_onejob.csv").write_text("")
pd.DataFrame([row]).to_csv("result/optimization_runs/round18_architecture/manifests/stage18c_smoke_onejob.csv", index=False)
print("wrote smoke onejob")
PY
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18c_smoke_onejob.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 --micro-batch-candidates 16,8 --limit 1
else
  EXTRA=()
  if [[ -n "${LIMIT_JOBS}" ]]; then EXTRA+=(--limit "${LIMIT_JOBS}"); fi
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18c_cross_attention_manifest.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 --micro-batch-candidates 512,256,128,64,32 "${EXTRA[@]}"
fi
echo "========== ROUND18 STAGE 18C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
