#!/usr/bin/env bash
# Round 4.1 target-to-source InfoNCE pretrain only (run INSIDE DAPL container)
set -euo pipefail
ROOT="/workspace/DAPL"
RUN_ID="vaewc_round4_1_t2s_infonce_collapse_guard"
RUN_DIR="${ROOT}/result/optimization_runs/${RUN_ID}"
SWEEP="${ROOT}/config/pretrain_sweeps/vaewc_round4_1_t2s_infonce_collapse_guard.json"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RUN_DIR}/pretrain" "${RUN_DIR}/selection"

exec > >(tee -a "${LOG_DIR}/round4_1_pretrain.log") 2>&1
echo "=== Round 4.1 pretrain start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "${ROOT}"

python3 tools/optimization_config_generator.py \
  --sweep-spec "${SWEEP}" --force

python3 tools/optimization_runner.py pretrain \
  --manifest "${RUN_DIR}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${RUN_DIR}" \
  --batch-size 128 --max-parallel 20

echo "=== Round 4.1 pretrain done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
