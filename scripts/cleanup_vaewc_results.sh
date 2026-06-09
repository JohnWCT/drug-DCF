#!/usr/bin/env bash
# Prune VAEwC result folders, keeping only selected experiments and summary files.
#
# Usage (inside DAPL container or via docker exec):
#   bash /workspace/DAPL/scripts/cleanup_vaewc_results.sh
#   bash /workspace/DAPL/scripts/cleanup_vaewc_results.sh --dry-run
#
# From host:
#   docker exec DAPL bash /workspace/DAPL/scripts/cleanup_vaewc_results.sh

set -euo pipefail

KEEP_EXPS=(exp_174 exp_746 exp_099)
TARGET_DIRS=(
  /workspace/DAPL/result/pretrain_vaewc
  /workspace/DAPL/result/pretrain_vaewc_loss_v2
)
DRY_RUN=false

info()  { printf '[INFO] %s\n' "$*"; }
warn()  { printf '[WARN] %s\n' "$*" >&2; }

is_kept_exp() {
  local exp_name="$1"
  local keep_id
  for keep_id in "${KEEP_EXPS[@]}"; do
    [[ "$exp_name" == "$keep_id" ]] && return 0
  done
  return 1
}

should_keep_root_file() {
  local file="$1"
  local base
  base="$(basename "$file")"
  case "$base" in
    *.csv|*.json) return 0 ;;
    *) return 1 ;;
  esac
}

prune_directory() {
  local base_dir="$1"
  local deleted=0
  local kept=0

  [[ -d "$base_dir" ]] || { warn "Skip missing directory: $base_dir"; return; }
  info "Processing: $base_dir"

  for entry in "$base_dir"/*; do
    [[ -e "$entry" ]] || continue
    local name
    name="$(basename "$entry")"

    if [[ -d "$entry" && "$name" =~ ^exp_ ]]; then
      if is_kept_exp "$name"; then
        info "  keep dir:  $name"
        kept=$((kept + 1))
      else
        if [[ "$DRY_RUN" == true ]]; then
          info "  [dry-run] would delete: $name"
        else
          rm -rf "$entry"
          info "  deleted:   $name"
        fi
        deleted=$((deleted + 1))
      fi
      continue
    fi

    if [[ -d "$entry" ]]; then
      info "  keep dir:  $name"
      continue
    fi

    if should_keep_root_file "$entry"; then
      info "  keep file: $name"
      continue
    fi

    if [[ "$DRY_RUN" == true ]]; then
      info "  [dry-run] would delete file: $name"
    else
      rm -f "$entry"
      info "  deleted file: $name"
    fi
    deleted=$((deleted + 1))
  done

  info "Summary for $base_dir: kept_exp=$kept, removed=$deleted"
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run) DRY_RUN=true; shift ;;
      -h|--help)
        sed -n '2,10p' "$0"
        exit 0
        ;;
      *) warn "Unknown argument: $1"; exit 1 ;;
    esac
  done

  info "Keeping experiments: ${KEEP_EXPS[*]}"
  info "Keeping subdirs like 00_report and root-level *.csv / *.json"
  [[ "$DRY_RUN" == true ]] && warn "DRY RUN — no files will be deleted"

  for dir in "${TARGET_DIRS[@]}"; do
    prune_directory "$dir"
  done

  info "Done."
}

main "$@"
