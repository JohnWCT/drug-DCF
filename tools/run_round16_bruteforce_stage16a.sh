#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=tools/run_round16_skip_downstream_guard.sh
source "$(dirname "$0")/run_round16_skip_downstream_guard.sh"
round16_skip_downstream_if_deferred "16A"

# shellcheck source=tools/run_round16_notify_helpers.sh
source "$(dirname "$0")/run_round16_notify_helpers.sh"

ROUND16_ROOT="result/optimization_runs/round16_bruteforce"
FINETUNE_PARALLEL="${FINETUNE_PARALLEL:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
FINETUNE_MINI_BATCH_SIZE="${FINETUNE_MINI_BATCH_SIZE:-3072}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1500}"

echo "========== ROUND16 STAGE 16A START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python3 tools/round16_bruteforce_config_builder.py \
  --settings config/round16_bruteforce_settings.json \
  --outdir "${ROUND16_ROOT}" \
  --stage 16a \
  --force

python3 tools/extract_round13_proto_features.py \
  --manifest "${ROUND16_ROOT}/manifests/proto_feature_manifest.csv" \
  --outdir "${ROUND16_ROOT}/features"

python3 tools/optimization_runner.py finetune \
  --manifest "${ROUND16_ROOT}/manifests/finetune_dispatch_manifest.csv" \
  --run-dir "${ROUND16_ROOT}" \
  --finetune-config config/params_finetune_round16_bruteforce.json \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --mini-batch-size "${FINETUNE_MINI_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --max-parallel "${FINETUNE_PARALLEL}" \
  --force-manifest \
  --round13-mode

python3 tools/optimization_runner.py aggregate \
  --run-dir "${ROUND16_ROOT}"

python3 tools/analyze_round16_bruteforce.py \
  --run-dir "${ROUND16_ROOT}" \
  --round13-root result/optimization_runs/round13_proto_response \
  --round15-root result/optimization_runs/round15_repro_rescue \
  --aggregate "${ROUND16_ROOT}/aggregate/aggregate_scores.csv" \
  --stage 16a \
  --outdir "${ROUND16_ROOT}/reports"

r16_notify --event stage-done --stage 16A
echo "========== ROUND16 STAGE 16A DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
