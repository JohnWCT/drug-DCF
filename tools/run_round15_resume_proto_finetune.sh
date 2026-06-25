#!/usr/bin/env bash
set -euo pipefail

ROUND15_ROOT="${ROUND15_ROOT:-result/optimization_runs/round15_repro_rescue}"

FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1000}"

echo "========== ROUND15 RESUME (own_plus_summary) START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

FT_MANIFEST="${ROUND15_ROOT}/manifests/finetune_dispatch_manifest.csv"
PROTO_MANIFEST="${ROUND15_ROOT}/manifests/proto_feature_manifest.csv"

echo "[Round15 resume] Re-extract prototype features (honor manifest combined_latent_dir)"
python3 tools/extract_round13_proto_features.py \
  --manifest "${PROTO_MANIFEST}" \
  --outdir "${ROUND15_ROOT}/features"

echo "[Round15 resume] Reset own_plus_summary finetune jobs to pending"
python3 - <<'PY'
import pandas as pd

path = "result/optimization_runs/round15_repro_rescue/manifests/finetune_dispatch_manifest.csv"
df = pd.read_csv(path)
mask = df["feature_mode"].astype(str) == "own_plus_summary"
n = int(mask.sum())
df.loc[mask, "status"] = "pending"
df.loc[mask, "start_time"] = ""
df.loc[mask, "end_time"] = ""
df.loc[mask, "error_message"] = ""
df.to_csv(path, index=False)
print(f"Reset {n} own_plus_summary jobs to pending")
PY

echo "[Round15 resume] Finetune own_plus_summary only"
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${ROUND15_ROOT}" \
  --finetune-config config/params_finetune_round15_compact_features.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

echo "[Round15 resume] Aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND15_ROOT}"

python3 tools/optimization_runner.py report \
  --run-dir "${ROUND15_ROOT}"

echo "[Round15 resume] Final analysis"
python3 tools/analyze_round15_repro_rescue.py \
  --run-dir "${ROUND15_ROOT}" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round14-root result/optimization_runs/round14_vicreg_stabilizer \
  --aggregate "${ROUND15_ROOT}/aggregate/aggregate_scores.csv" \
  --outdir "${ROUND15_ROOT}/final_report"

echo "========== ROUND15 RESUME DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
