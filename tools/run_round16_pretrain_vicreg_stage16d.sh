#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
STAGE_ROOT="${ROUND16_ROOT}/stage16d"
PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-12}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"

echo "========== ROUND16 STAGE 16D START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

enabled="$(python3 - <<'PY'
import json
with open("config/round16_bruteforce_settings.json", encoding="utf-8") as f:
    cfg = json.load(f)
print("true" if cfg.get("stage16d", {}).get("enabled", False) else "false")
PY
)"

if [[ "${enabled}" != "true" ]]; then
  echo "Stage 16D disabled in config — skip."
  r16_notify --event stage-done --stage 16D
  echo "========== ROUND16 STAGE 16D SKIPPED $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  exit 0
fi

r16_notify --event stage-start --stage 16D

python3 tools/round16_bruteforce_config_builder.py \
  --settings config/round16_bruteforce_settings.json \
  --outdir "${ROUND16_ROOT}" \
  --stage 16d \
  --force

python3 tools/optimization_runner.py pretrain \
  --manifest "${STAGE_ROOT}/manifests/stage16d_pretrain_manifest.csv" \
  --run-dir "${STAGE_ROOT}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_PARALLEL}"

python3 - <<'PY'
import glob
import json
import os
import pandas as pd

stage_root = "result/optimization_runs/round16_bruteforce/stage16d"
rows = []
for path in sorted(glob.glob(os.path.join(stage_root, "pretrain", "exp_*", "run_summary.json"))):
    exp_id = os.path.basename(os.path.dirname(path))
    payload = json.load(open(path, encoding="utf-8"))
    params = payload.get("params", {})
    metrics = payload.get("metrics", {})
    rows.append(
        {
            "exp_id": exp_id,
            "status": payload.get("status", "unknown"),
            "round16_lineage": params.get("round16_lineage", ""),
            "lambda_tumor_var": params.get("lambda_tumor_var"),
            "lambda_tumor_cov": params.get("lambda_tumor_cov"),
            "tumor_vicreg_start_epoch": params.get("tumor_vicreg_start_epoch"),
            "tumor_vicreg_full_epoch": params.get("tumor_vicreg_full_epoch"),
            "random_seed": params.get("random_seed"),
            "kmeans_ari": metrics.get("kmeans_ari"),
            "wasserstein": metrics.get("wasserstein"),
            "fid": metrics.get("fid"),
        }
    )
outdir = os.path.join(stage_root, "reports")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "stage16d_pretrain_summary.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {out} ({len(rows)} rows)")
PY

r16_notify --event stage-done --stage 16D
echo "========== ROUND16 STAGE 16D DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
