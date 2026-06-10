#!/usr/bin/env bash
# Wait for Round 3 pretrain to finish, then run select → finetune → aggregate.
set -euo pipefail
cd /workspace/DAPL

RUN_DIR="${RUN_DIR:-result/optimization_runs/vaewc_proto_infonce_round3_exp746}"
MANIFEST="${RUN_DIR}/manifests/pretrain_sweep_manifest.csv"
TOP10="${RUN_DIR}/selection/pretrain_top10.csv"
FT_MANIFEST="${RUN_DIR}/manifests/finetune_dispatch_manifest.csv"
LOG="${RUN_DIR}/logs/round3_post_pretrain.log"

FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4096}"
FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-1024}"
FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-26}"
MIN_PASSING="${MIN_PASSING:-10}"
REQUIRE_CONTROLS="${REQUIRE_CONTROLS:-2}"

mkdir -p "${RUN_DIR}/logs"
exec >> "${LOG}" 2>&1

echo "========== POST-PRETRAIN WATCHER $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

while true; do
  pending=$(python3 - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/vaewc_proto_infonce_round3_exp746/manifests/pretrain_sweep_manifest.csv")
print(int((df["status"].isin(["pending", "running"])).sum()))
PY
)
  if [ "${pending}" -eq 0 ]; then
    break
  fi
  sleep 120
done

python3 - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/vaewc_proto_infonce_round3_exp746/manifests/pretrain_sweep_manifest.csv")
print("Pretrain final:", df["status"].value_counts().to_dict())
PY

echo "=== Stage 2: Selection ==="
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

echo "=== Stage 3+4 ==="
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

echo "========== POST-PRETRAIN WATCHER END $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
