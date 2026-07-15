#!/usr/bin/env bash
# Safe Round 19G executor runner. Default smoke never starts formal inference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODE="${1:-smoke}"
STATUS_DIR="${ROUND19G_STATUS_DIR:-result/optimization_runs/round19_factorial/stage19g/status}"
mkdir -p "${STATUS_DIR}"
STATUS="${STATUS_DIR}/${MODE}.json"
notify() { python3 tools/round19_telegram_notify.py "$@" || true; }
fail() {
  python3 -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"failed","mode":sys.argv[2],"reason":sys.argv[3]},indent=2)+"\n")' "${STATUS}" "${MODE}" "$1"
  notify --event stage-fail --stage "19g-${MODE}" --reason "$1"
}
trap 'fail "exit ${?} at line ${LINENO}"' ERR

if [[ -f "${STATUS}" ]] && python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="done" else 1)' "${STATUS}"; then
  echo "19G ${MODE} already done; resume skipped"
  exit 0
fi
notify --event stage-start --stage "19g-${MODE}"
python3 -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"running","mode":sys.argv[2]},indent=2)+"\n")' "${STATUS}" "${MODE}"

case "${MODE}" in
  smoke)
    bash tools/run_round19_stage19g_setup_smoke.sh
    ;;
  pilot)
    : "${ROUND19G_EXPERIMENT_LOCK:?Set immutable 19G experiment lock path}"
    python3 tools/round19_stage19g_dispatch.py --pilot \
      --experiment-lock "${ROUND19G_EXPERIMENT_LOCK}" \
      --output-root "${ROUND19G_OUTPUT_DIR:-result/optimization_runs/round19_factorial/stage19g_pilot}" \
      --execute
    ;;
  formal)
    if [[ "${ROUND19G_ALLOW_FORMAL:-0}" != "1" ]]; then
      echo "Formal execution blocked; set ROUND19G_ALLOW_FORMAL=1 only after locks/cases are finalized." >&2
      exit 2
    fi
    : "${ROUND19G_EXPERIMENT_LOCK:?Set immutable 19G experiment lock path}"
    python3 tools/round19_stage19g_dispatch.py --formal \
      --experiment-lock "${ROUND19G_EXPERIMENT_LOCK}" \
      --output-root "${ROUND19G_OUTPUT_DIR:-result/optimization_runs/round19_factorial/stage19g}" \
      --execute
    ;;
  *)
    echo "usage: $0 [smoke|pilot|formal]" >&2
    exit 2
    ;;
esac

python3 -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"done","mode":sys.argv[2]},indent=2)+"\n")' "${STATUS}" "${MODE}"
trap - ERR
notify --event stage-done --stage "19g-${MODE}"
