#!/usr/bin/env bash
# Round 6 multi-branch pretrain (6A–6E).
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
PARALLEL="${PARALLEL:-20}"

for SPEC in \
  vaewc_round6A_tumor_topology \
  vaewc_round6B_topology_classgap_combo \
  vaewc_round6C_tumor_transfer_subspace \
  vaewc_round6D_within_domain_tumor_supcon \
  vaewc_round6E_tumor_vicreg_stabilizer
do
  RUN="result/optimization_runs/${SPEC}"
  echo "=== generate ${SPEC} ==="
  python3 tools/optimization_runner.py generate \
    --sweep-spec "config/pretrain_sweeps/${SPEC}.json" \
    --run-dir "${RUN}" \
    --force
  echo "=== pretrain ${SPEC} (parallel=${PARALLEL}) ==="
  python3 tools/optimization_runner.py pretrain \
    --manifest "${RUN}/manifests/pretrain_sweep_manifest.csv" \
    --run-dir "${RUN}" \
    --device "${DEVICE}" \
    --max-parallel "${PARALLEL}"
done

echo "=== diagnostics ==="
python3 tools/analyze_round6_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round6A_tumor_topology \
    result/optimization_runs/vaewc_round6B_topology_classgap_combo \
    result/optimization_runs/vaewc_round6C_tumor_transfer_subspace \
    result/optimization_runs/vaewc_round6D_within_domain_tumor_supcon \
    result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer \
  --out-dir result/optimization_runs/round6_combined/reports
