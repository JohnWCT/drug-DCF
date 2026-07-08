#!/usr/bin/env bash
# Invalidate Round 17 artifacts built with 28-class prototype cache; keep pretrain checkpoints.
set -euo pipefail

ROUND17_ROOT="${ROUND17_ROOT:-result/optimization_runs/round17_direct_proto}"
STAMP="${STAMP:-pre18class_fix_$(date -u +%Y%m%dT%H%M%SZ)}"

echo "========== ROUND17 18-CLASS RERUN PREP (${STAMP}) =========="
echo "ROUND17_ROOT=${ROUND17_ROOT}"

for stage in reports_stage17a reports_stage17b reports_stage17c; do
  src="${ROUND17_ROOT}/${stage}"
  if [[ -d "${src}" ]]; then
    dst="${ROUND17_ROOT}/${stage}_${STAMP}"
    if [[ ! -e "${dst}" ]]; then
      mv "${src}" "${dst}"
      echo "archived ${src} -> ${dst}"
    else
      echo "skip archive ${src}: ${dst} exists"
    fi
  fi
done

echo "removing stale prototype caches under features/"
find "${ROUND17_ROOT}/features" -type d -name '_proto_cache' -prune -exec rm -rf {} + 2>/dev/null || true

echo "prep complete"
