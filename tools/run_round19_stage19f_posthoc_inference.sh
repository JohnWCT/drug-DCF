#!/usr/bin/env bash
# Round 19F final-lock-only post-hoc manifest build and inference orchestration.
# Safe default: verify/build manifests and dry-run dispatch; no inference jobs start.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
LOCK="${OUTDIR}/reports/round19_final_role_lock.json"
INTERNAL_MANIFEST="${OUTDIR}/manifests/stage19f_posthoc_internal_test_manifest.csv"
TCGA_MANIFEST="${OUTDIR}/manifests/stage19f_posthoc_tcga_manifest.csv"
PIPELINE="${ROUND19_PIPELINE:-step1_finetune_latent_pipeline_round19.py}"
TARGET_VRAM_FRACTION="${ROUND19_TARGET_VRAM_FRACTION:-0.90}"
ESTIMATED_JOB_MB="${ROUND19_INFER_ESTIMATED_JOB_MB:-3500}"
LIMIT_JOBS="${LIMIT_JOBS:-}"
EXECUTE=0

usage() {
  echo "Usage: $0 [--execute]"
  echo "Without --execute, manifests are built and both dispatches are dry runs."
}

for arg in "$@"; do
  case "${arg}" in
    --execute) EXECUTE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${LOCK}" ]]; then
  echo "Missing immutable final lock: ${LOCK}" >&2
  exit 3
fi

python3 tools/round19_stage19f_posthoc_manifest.py \
  --final-lock "${LOCK}" \
  --project-root "${ROOT}" \
  --output-root "${OUTDIR}/stage19f_posthoc" \
  --internal-output "${INTERNAL_MANIFEST}" \
  --tcga-output "${TCGA_MANIFEST}"

COMMON_ARGS=(
  --final-lock "${LOCK}"
  --project-root "${ROOT}"
  --pipeline "${PIPELINE}"
  --target-vram-fraction "${TARGET_VRAM_FRACTION}"
  --estimated-job-mb "${ESTIMATED_JOB_MB}"
)
if [[ -n "${LIMIT_JOBS}" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT_JOBS}")
fi

if [[ "${EXECUTE}" != "1" ]]; then
  echo "[19F post-hoc] Safe dry run; no inference process will start."
  python3 tools/round19_stage19f_inference_dispatch.py \
    --manifest "${INTERNAL_MANIFEST}" "${COMMON_ARGS[@]}"
  python3 tools/round19_stage19f_inference_dispatch.py \
    --manifest "${TCGA_MANIFEST}" "${COMMON_ARGS[@]}"
  exit 0
fi

r19_notify() {
  python3 tools/round19_telegram_notify.py "$@" || true
}
failed() {
  rc=$?
  r19_notify --event stage-fail --stage 19f-posthoc --reason "exit_${rc}"
  exit "${rc}"
}
trap failed ERR

r19_notify --event stage-start --stage 19f-posthoc \
  --extra "final-lock-only internal=90 TCGA=450; target VRAM=${TARGET_VRAM_FRACTION}"

python3 tools/round19_stage19f_inference_dispatch.py \
  --manifest "${INTERNAL_MANIFEST}" "${COMMON_ARGS[@]}" --execute
r19_notify --event progress --stage 19f-posthoc --manifest "${INTERNAL_MANIFEST}"

python3 tools/round19_stage19f_inference_dispatch.py \
  --manifest "${TCGA_MANIFEST}" "${COMMON_ARGS[@]}" --execute
r19_notify --event progress --stage 19f-posthoc --manifest "${TCGA_MANIFEST}"

r19_notify --event stage-done --stage 19f-posthoc --manifest "${TCGA_MANIFEST}"
trap - ERR
echo "Round 19F post-hoc inference completed."
