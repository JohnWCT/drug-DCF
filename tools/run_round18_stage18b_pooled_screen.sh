#!/usr/bin/env bash
# Round 18 Stage 18B: build pooled screening manifest + optional smoke
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-1}"

echo "========== ROUND18 STAGE 18B POOLED SCREEN START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "SMOKE_ONLY=${SMOKE_ONLY}"

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
print(f"18B manifest OK: {len(df)} jobs, families={sorted(families)}")
print(df.groupby(["architecture_family","omics_mode"]).size().to_string())
PY

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  echo "[18B] running synthetic pipeline smoke (full 45-job screen not launched)"
  python step1_finetune_latent_pipeline_round18_cv.py --mode smoke --outdir "${OUTDIR}" --steps 2
else
  echo "[18B] SMOKE_ONLY=0: full screening runner not yet wired; refusing to no-op train 45 jobs"
  exit 2
fi

echo "========== ROUND18 STAGE 18B POOLED SCREEN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
