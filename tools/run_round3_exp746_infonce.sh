#!/usr/bin/env bash
# Round 3: exp_746 baseline (lambda_cls=20) + gentle InfoNCE + strict K-means filter.
set -euo pipefail
cd /workspace/DAPL

RUN_DIR="${RUN_DIR:-result/optimization_runs/vaewc_proto_infonce_round3_exp746}"
SWEEP="${SWEEP:-config/pretrain_sweeps/vaewc_proto_infonce_round3_exp746.json}"
MANIFEST="${RUN_DIR}/manifests/pretrain_sweep_manifest.csv"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round3_exp746_pipeline.log"

# CCLE ~1128 rows: batch must stay <1128 (128 recommended; do NOT override to 2048+)
# batch=128 uses ~0.8GB/job → parallel=20 ≈ 50% GPU mem (vs parallel=4 ≈ 7%)
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_MAX_PARALLEL="${PRETRAIN_MAX_PARALLEL:-20}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-26}"
MIN_PASSING="${MIN_PASSING:-10}"
REQUIRE_CONTROLS="${REQUIRE_CONTROLS:-2}"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "========== ROUND3 exp746+InfoNCE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "baseline=exp_746 lambda_cls=20 batch=${PRETRAIN_BATCH_SIZE} parallel=${PRETRAIN_MAX_PARALLEL}"

python3 tools/optimization_config_generator.py --sweep-spec "${SWEEP}" --force
python3 tools/update_running_report.py --run-dir "${RUN_DIR}" --note "Round3: exp_746 baseline + InfoNCE sweep; filter gate enabled."

echo "=== Stage 1: Pretrain (${SWEEP}) ==="
python3 tools/optimization_runner.py pretrain \
  --manifest "${MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --max-parallel "${PRETRAIN_MAX_PARALLEL}"

echo "=== Stage 2: Selection (strict filter) ==="
set +e
python3 tools/optimization_runner.py select \
  --run-dir "${RUN_DIR}" \
  --filter-config config/visualize_vaewc_filter.json \
  --min-passing "${MIN_PASSING}" \
  --require-controls "${REQUIRE_CONTROLS}"
SELECT_RC=$?
set -e
if [ "${SELECT_RC}" -ne 0 ]; then
  echo ">>> Filter gate failed (exit ${SELECT_RC}). See ${RUN_DIR}/selection/filter_threshold_report.csv"
  python3 tools/update_running_report.py --run-dir "${RUN_DIR}" --note "Insufficient filter pass; tune sweep and rerun failed jobs."
  exit "${SELECT_RC}"
fi

echo "=== Stage 3+4: Finetune + aggregate ==="
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" --run-dir "${RUN_DIR}" --top10 "${TOP10}" \
  --build-manifest-only --force-manifest
python3 tools/optimization_runner.py finetune \
  --manifest "${FT_MANIFEST}" --run-dir "${RUN_DIR}" --top10 "${TOP10}" \
  --epochs 1000 --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH}" --max-parallel "${FINETUNE_MAX_PARALLEL}"
python3 tools/optimization_runner.py aggregate --run-dir "${RUN_DIR}"
python3 tools/optimization_runner.py report --run-dir "${RUN_DIR}"
python3 tools/update_running_report.py --run-dir "${RUN_DIR}" --note "Round3 complete."

echo "========== ROUND3 END $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
