#!/usr/bin/env bash
# Run Round 17R stages C -> D -> F sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-20}"

bash "${SCRIPT_DIR}/run_round17r_stage17r_c_refine.sh"
bash "${SCRIPT_DIR}/run_round17r_stage17r_d_confirm.sh"
bash "${SCRIPT_DIR}/run_round17r_stage17r_f_tsne.sh"

echo "========== ROUND17R STAGES 17R-C/D/F ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
