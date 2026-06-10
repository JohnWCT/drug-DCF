#!/usr/bin/env bash
# Round 3: relaxed filter → Top-10 selection → parallel finetune → aggregate.
set -euo pipefail
cd /workspace/DAPL

RUN_DIR="${RUN_DIR:-result/optimization_runs/vaewc_proto_infonce_round3_exp746}"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round3_finetune_relaxed.log"
FILTER_CONFIG="${FILTER_CONFIG:-config/visualize_vaewc_filter.json}"

FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-26}"
MIN_PASSING="${MIN_PASSING:-10}"
REQUIRE_CONTROLS="${REQUIRE_CONTROLS:-2}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND3 RELAXED FILTER + FINETUNE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "filter=${FILTER_CONFIG} (fid<=30, wasserstein<=0.70; kmeans unchanged)"

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Relaxed fid/wasserstein filter; starting selection + finetune validation."

echo "=== Stage 2: Selection ==="
python3 tools/optimization_runner.py select \
  --run-dir "${RUN_DIR}" \
  --filter-config "${FILTER_CONFIG}" \
  --min-passing "${MIN_PASSING}" \
  --require-controls "${REQUIRE_CONTROLS}"

echo "=== Stage 3: Finetune (parallel=${FINETUNE_MAX_PARALLEL}) ==="
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --top10 "${TOP10}" \
  --build-manifest-only \
  --force-manifest

python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --top10 "${TOP10}" \
  --epochs 1000 \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH}" \
  --max-parallel "${FINETUNE_MAX_PARALLEL}"

echo "=== Stage 4: Aggregate + report ==="
python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"
python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Round3 finetune validation complete (relaxed filter)."

echo "========== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
