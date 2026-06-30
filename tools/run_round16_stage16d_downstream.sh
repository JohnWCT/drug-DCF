#!/usr/bin/env bash
# Stage 16D: filter pretrain -> downstream finetune -> analyze + architecture summary.
set -euo pipefail

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

export FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-24576}"
export FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-6144}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
STAGE_ROOT="${ROUND16_ROOT}/stage16d"
DOWNSTREAM_RUN="${STAGE_ROOT}/downstream"
LOG="logs/round16_stage16d_downstream.log"

mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

TOP_K="$(python3 - <<'PY'
import json
with open("config/round16_bruteforce_settings.json", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("stage16d", {}).get("downstream", {}).get("top_k_per_lineage", 4))
PY
)"

log "========== 16D FILTER + DOWNSTREAM =========="

log "Step 1: filter pretrain candidates (top ${TOP_K}/lineage)"
python3 tools/round16_stage16d_selection.py \
  --stage-root "${STAGE_ROOT}" \
  --top-k-per-lineage "${TOP_K}"

log "Step 2: build downstream manifests"
python3 tools/round16_stage16d_downstream_builder.py \
  --settings config/round16_bruteforce_settings.json \
  --stage-root "${STAGE_ROOT}" \
  --force

r16_notify --event stage-start --stage 16D

log "Step 3: feature extraction"
python3 tools/extract_round13_proto_features.py \
  --manifest "${STAGE_ROOT}/manifests/stage16d_downstream_proto_feature_manifest.csv" \
  --outdir "${STAGE_ROOT}/features_downstream"

log "Step 4: downstream finetune (parallel=${FINETUNE_PARALLEL})"
python3 tools/optimization_runner.py finetune \
  --manifest "${STAGE_ROOT}/manifests/stage16d_downstream_finetune_manifest.csv" \
  --run-dir "${DOWNSTREAM_RUN}" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode 2>&1 | tee -a logs/round16_16d_downstream_finetune.log

log "Step 5: aggregate"
python3 tools/optimization_runner.py aggregate \
  --run-dir "${DOWNSTREAM_RUN}"

log "Step 6: analyze + architecture summary"
python3 tools/analyze_round16_stage16d.py \
  --stage-root "${STAGE_ROOT}" \
  --outdir "${STAGE_ROOT}/reports"

r16_notify --event stage-done --stage 16D
log "========== 16D DOWNSTREAM DONE =========="
