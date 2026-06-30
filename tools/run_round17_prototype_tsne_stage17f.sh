#!/usr/bin/env bash
set -euo pipefail

ROUND17_ROOT="${ROUND17_ROOT:-result/optimization_runs/round17_direct_proto}"
TSNE_PARALLEL="${TSNE_PARALLEL:-1}"

echo "========== ROUND17 STAGE 17F PROTOTYPE TSNE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/visualize_round17_prototype_tsne.py \
  --settings config/round17_direct_proto_settings.json \
  --manifest "${ROUND17_ROOT}/manifests/stage17a_proto_feature_manifest.csv" \
  --outdir "${ROUND17_ROOT}/visualizations/prototype_tsne" \
  --models r13_exp_008 round16_top1 \
  --force

echo "========== ROUND17 STAGE 17F PROTOTYPE TSNE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
