#!/usr/bin/env bash
# Round 4 cross-domain InfoNCE full pipeline (run INSIDE DAPL container)
set -euo pipefail
ROOT="/workspace/DAPL"
RUN_ID="vaewc_round4_cross_domain_infonce"
RUN_DIR="${ROOT}/result/optimization_runs/${RUN_ID}"
SWEEP="${ROOT}/config/pretrain_sweeps/vaewc_round4_cross_domain_infonce.json"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RUN_DIR}/pretrain" "${RUN_DIR}/selection"

exec > >(tee -a "${LOG_DIR}/round4_full_pipeline.log") 2>&1
echo "=== Round 4 pipeline start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "${ROOT}"

python3 tools/optimization_config_generator.py \
  --sweep-spec "${SWEEP}" --force

python3 tools/optimization_runner.py pretrain \
  --manifest "${RUN_DIR}/manifests/pretrain_sweep_manifest.csv" \
  --run-dir "${RUN_DIR}" \
  --batch-size 128 --max-parallel 20

python3 tools/optimization_runner.py select \
  --run-dir "${RUN_DIR}" \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round4_kmeans_first \
  --exclude-proto-ineffective \
  --min-passing 10 --require-controls 2 \
  --run-tag "${RUN_ID}"

python3 tools/optimization_runner.py finetune \
  --manifest "${RUN_DIR}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${RUN_DIR}" \
  --top10 "${RUN_DIR}/selection/pretrain_top10.csv" \
  --epochs 1000 --batch-size 4096 --mini-batch-size 1024 --max-parallel 26

python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" || true

echo "=== Round 4 pipeline done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
