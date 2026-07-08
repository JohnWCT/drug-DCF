#!/usr/bin/env bash
set -euo pipefail

ROUND17R_ROOT="${ROUND17R_ROOT:-result/optimization_runs/round17r_18class}"
SETTINGS="${ROUND17R_SETTINGS:-config/round17r_18class_focused_settings.json}"
TSNE_OUTDIR="${ROUND17R_ROOT}/visualizations/prototype_tsne"
MANIFEST="${ROUND17R_ROOT}/manifests/stage17r_a_proto_feature_manifest.csv"

echo "========== ROUND17R STAGE 17R-F TSNE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

if [[ ! -f "${MANIFEST}" ]]; then
  python3 tools/round17r_18class_config_builder.py \
    --settings "${SETTINGS}" \
    --outdir "${ROUND17R_ROOT}" \
    --stage 17r_a
fi

python3 tools/visualize_round17_prototype_tsne.py \
  --settings "${SETTINGS}" \
  --manifest "${MANIFEST}" \
  --outdir "${TSNE_OUTDIR}" \
  --models r13_exp_008 r13_exp_035_control \
  --force

python3 - <<'PY'
import pandas as pd
from pathlib import Path
import os
root = Path(os.environ.get("ROUND17R_ROOT", "result/optimization_runs/round17r_18class")) / "visualizations" / "prototype_tsne"
for model in ["r13_exp_008", "r13_exp_035_control"]:
    csv = root / model / "prototype_tsne_coordinates.csv"
    if not csv.is_file():
        print(f"skip missing {csv}")
        continue
    df = pd.read_csv(csv)
    n_src = int((df["point_type"] == "source_prototype").sum())
    n_tgt = int((df["point_type"] == "target_prototype").sum())
    forbidden = {"Engineered", "Fibroblast", "Bone Cancer"}
    present = set(df["cancer_type"].dropna().astype(str)) & forbidden
    assert n_src == 18, (model, n_src)
    assert n_tgt == 18, (model, n_tgt)
    assert not present, (model, present)
    print(f"OK {model}: source={n_src} target={n_tgt}")
PY

echo "========== ROUND17R STAGE 17R-F TSNE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
