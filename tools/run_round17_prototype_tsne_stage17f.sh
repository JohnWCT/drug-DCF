#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=tools/run_round17_notify_helpers.sh
source "$(dirname "$0")/run_round17_notify_helpers.sh"

ROUND17_ROOT="${ROUND17_ROOT:-result/optimization_runs/round17_direct_proto}"
TSNE_OUTDIR="${ROUND17_ROOT}/visualizations/prototype_tsne"

echo "========== ROUND17 STAGE 17F PROTOTYPE TSNE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
r17_notify --event stage-start --stage 17F

python3 tools/visualize_round17_prototype_tsne.py \
  --settings config/round17_direct_proto_settings.json \
  --manifest "${ROUND17_ROOT}/manifests/stage17a_proto_feature_manifest.csv" \
  --outdir "${TSNE_OUTDIR}" \
  --models r13_exp_008 r13_exp_035_control \
  --force

python3 tools/round17_telegram_notify.py --event stage17f-done --outdir "${TSNE_OUTDIR}"
echo "========== ROUND17 STAGE 17F PROTOTYPE TSNE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
