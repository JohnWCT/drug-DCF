#!/usr/bin/env bash
set -euo pipefail

ROUND17R_ROOT="${ROUND17R_ROOT:-result/optimization_runs/round17r_18class}"
SETTINGS="${ROUND17R_SETTINGS:-config/round17r_18class_focused_settings.json}"

echo "========== ROUND17R STAGE 17R-A FEATURE SMOKE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/round17r_18class_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${ROUND17R_ROOT}" \
  --stage 17r_a

python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND17R_ROOT}/manifests/stage17r_a_proto_feature_manifest.csv" \
  --outdir "${ROUND17R_ROOT}/features" \
  --strict

python3 - <<'PY'
import json
from pathlib import Path

root = Path("result/optimization_runs/round17r_18class/features")
# also honor ROUND17R_ROOT via env if present
import os
root = Path(os.environ.get("ROUND17R_ROOT", "result/optimization_runs/round17r_18class")) / "features"
paths = list(root.rglob("feature_metadata.json"))
assert paths, f"No feature_metadata.json under {root}"
for p in paths:
    meta = json.loads(p.read_text())
    assert int(meta["n_trainable_cancer_types"]) == 18, p
    assert meta["uses_legacy_28class_cache"] is False, p
    assert meta.get("prototype_class_source") == "checkpoint_metadata", p
print(f"OK: {len(paths)} features are 18-class-clean")
PY

python3 tools/analyze_round17r_18class.py \
  --run-dir "${ROUND17R_ROOT}" \
  --settings "${SETTINGS}" \
  --stage 17r_a \
  --outdir "${ROUND17R_ROOT}/reports_stage17r_a"

echo "========== ROUND17R STAGE 17R-A FEATURE SMOKE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
