#!/usr/bin/env bash
# Reset manifest running→pending, then resume pretrain with gpu_parallel_profile defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/gpu_parallel_env.sh
source tools/gpu_parallel_env.sh

RUN_DIR="${1:?usage: resume_pretrain_with_profile.sh <run_dir> [manifest_name]}"
MANIFEST_NAME="${2:-pretrain_sweep_manifest.csv}"
MANIFEST="${RUN_DIR}/manifests/${MANIFEST_NAME}"
PARALLEL="${PRETRAIN_PARALLEL}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

python3 - "${MANIFEST}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
n = int((df["status"] == "running").sum())
if n:
    df.loc[df["status"] == "running", "status"] = "pending"
    df.to_csv(path, index=False)
    print(f"[resume] reset {n} running job(s) to pending in {path}")
else:
    print(f"[resume] no running jobs in {path}")
PY

echo "[resume] ${RUN_DIR} max_parallel=${PARALLEL}"
python3 tools/optimization_runner.py pretrain \
  --manifest "${MANIFEST}" \
  --run-dir "${RUN_DIR}" \
  --device "${DEVICE:-cuda}" \
  --max-parallel "${PARALLEL}"
