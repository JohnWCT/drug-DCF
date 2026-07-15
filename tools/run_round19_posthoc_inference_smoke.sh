#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${ROUND19_DOCKER_IMAGE:-}" && "${ROUND19_SMOKE_IN_CONTAINER:-0}" != "1" ]]; then
  docker run --rm \
    -e ROUND19_SMOKE_IN_CONTAINER=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$PWD:/workspace/DAPL" \
    -w /workspace/DAPL \
    "${ROUND19_DOCKER_IMAGE}" \
    bash tools/run_round19_posthoc_inference_smoke.sh
  exit $?
fi

python3 - <<'PY'
import ast
from pathlib import Path

for name in (
    "step1_finetune_latent_pipeline_round19.py",
    "tools/round19_dataset.py",
    "tools/round19_posthoc_inference_smoke.py",
):
    ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)
PY
python3 tools/round19_posthoc_inference_smoke.py

echo "Round19 post-hoc Docker smoke OK (synthetic only; no 540-job dispatch)"
