#!/usr/bin/env bash
# Source GPU parallelism defaults from config/gpu_parallel_profile.json.
# Override any value via environment before sourcing, e.g. PRETRAIN_PARALLEL=30.
set -euo pipefail

_PROFILE="${DAPL_GPU_PROFILE:-config/gpu_parallel_profile.json}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
_PROFILE_PATH="${_PROJECT_ROOT}/${_PROFILE}"

if [[ ! -f "${_PROFILE_PATH}" ]]; then
  echo "[gpu_parallel_env] missing profile: ${_PROFILE_PATH}" >&2
  export PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-36}"
  export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
  export FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-42}"
  export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12288}"
  export FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-3072}"
  return 0 2>/dev/null || exit 0
fi

_read_profile() {
  python3 - "${_PROFILE_PATH}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
pre = p.get("pretrain", {})
ft = p.get("finetune", {})
print(f"PRETRAIN_PARALLEL={pre.get('max_parallel', 36)}")
print(f"PRETRAIN_BATCH_SIZE={pre.get('batch_size', 128)}")
print(f"FINETUNE_MAX_PARALLEL={ft.get('max_parallel', 42)}")
print(f"FINETUNE_BATCH_SIZE={ft.get('batch_size', 12288)}")
print(f"FINETUNE_MINI_BATCH={ft.get('mini_batch_size', 3072)}")
PY
}

while IFS='=' read -r key value; do
  case "${key}" in
    PRETRAIN_PARALLEL) export PRETRAIN_PARALLEL="${PRETRAIN_PARALLEL:-${value}}" ;;
    PRETRAIN_BATCH_SIZE) export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-${value}}" ;;
    FINETUNE_MAX_PARALLEL) export FINETUNE_MAX_PARALLEL="${FINETUNE_MAX_PARALLEL:-${value}}" ;;
    FINETUNE_BATCH_SIZE) export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-${value}}" ;;
    FINETUNE_MINI_BATCH) export FINETUNE_MINI_BATCH="${FINETUNE_MINI_BATCH:-${value}}" ;;
  esac
done < <(_read_profile)

# Aliases used by some Round 6 scripts.
export PARALLEL="${PARALLEL:-${PRETRAIN_PARALLEL}}"
export PRETRAIN_MAX_PARALLEL="${PRETRAIN_MAX_PARALLEL:-${PRETRAIN_PARALLEL}}"
