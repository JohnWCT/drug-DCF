#!/usr/bin/env bash
# Exit 0 when downstream brute-force stages (16A–16C) are deferred to a later round.

round16_skip_downstream_if_deferred() {
  local stage_label="$1"
  if [[ "${ROUND16_SKIP_DOWNSTREAM:-0}" == "1" ]] || [[ -f config/round16_defer_downstream.flag ]]; then
    echo "========== ROUND16 STAGE ${stage_label} SKIPPED (deferred to next round) =========="
    echo "Remove config/round16_defer_downstream.flag or run tools/run_round16_downstream_sweep.sh when ready."
    exit 0
  fi
}
