#!/usr/bin/env bash
# Resume Round 3 pretrain with higher parallel (after batch_size drop to 128).
set -euo pipefail
cd /workspace/DAPL

RUN_DIR="${RUN_DIR:-result/optimization_runs/vaewc_proto_infonce_round3_exp746}"
MANIFEST="${RUN_DIR}/manifests/pretrain_sweep_manifest.csv"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round3_exp746_pipeline.log"

PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_MAX_PARALLEL="${PRETRAIN_MAX_PARALLEL:-20}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-26}"
MIN_PASSING="${MIN_PASSING:-10}"
REQUIRE_CONTROLS="${REQUIRE_CONTROLS:-2}"

mkdir -p "${RUN_DIR}/logs"

echo "Stopping prior Round 3 runners..."
pkill -f 'run_round3_exp746_infonce.sh' 2>/dev/null || true
pkill -f 'optimization_runner.py pretrain' 2>/dev/null || true
pkill -f 'pretrain_VAEwC.py.*vaewc_proto_infonce_round3_exp746' 2>/dev/null || true
sleep 5

exec >> "${LOG}" 2>&1
echo ""
echo "========== ROUND3 RESUME $(date -u +%Y-%m-%dT%H:%M:%SZ) parallel=${PRETRAIN_MAX_PARALLEL} batch=${PRETRAIN_BATCH_SIZE} =========="

python3 - <<'PY'
import pandas as pd
p = "result/optimization_runs/vaewc_proto_infonce_round3_exp746/manifests/pretrain_sweep_manifest.csv"
df = pd.read_csv(p)
df.loc[df["status"] == "running", "status"] = "pending"
df.to_csv(p, index=False)
print(df["status"].value_counts().to_dict())
PY

python3 tools/update_running_report.py --run-dir "${RUN_DIR}" \
  --note "Resumed pretrain with max_parallel=${PRETRAIN_MAX_PARALLEL} (batch=${PRETRAIN_BATCH_SIZE})."

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
  echo ">>> Filter gate failed (exit ${SELECT_RC})."
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

echo "========== ROUND3 RESUME END $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
