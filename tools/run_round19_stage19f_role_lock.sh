#!/usr/bin/env bash
# Round 19F proposal-review gate. This script never creates the final role lock.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
POLICY="${ROUND19_STAGE19F_POLICY:-config/round19_stage19f_role_policy.json}"
PROPOSAL="${OUTDIR}/reports/round19_final_role_proposal.json"
INVENTORY="${OUTDIR}/reports/round19_stage19f_checkpoint_inventory.csv"
SUMMARY="${OUTDIR}/reports/round19_stage19f_checkpoint_inventory_summary.json"
ROLE_LOCK="${OUTDIR}/reports/round19_final_role_lock.json"
MANIFEST_19D="${OUTDIR}/manifests/stage19d_manifest.csv"

r19_notify() { python3 tools/round19_telegram_notify.py "$@" || true; }

SMOKE_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --smoke-only) SMOKE_ONLY=1 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -f "${ROLE_LOCK}" ]]; then
  echo "Refusing proposal regeneration after final role lock exists: ${ROLE_LOCK}" >&2
  exit 3
fi

mkdir -p "${OUTDIR}/reports" logs
trap 'rc=$?; if [[ $rc -ne 0 ]]; then r19_notify --event stage-fail --stage 19f-proposal --reason "exit_${rc}"; fi' EXIT

r19_notify --event stage-start --stage 19f-proposal \
  --extra "proposal-review gate; no internal/TCGA inference; no final role lock"

echo "========== ROUND19 STAGE 19F PROPOSAL $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "[19F] Build full-precision role proposal"
python3 tools/round19_stage19f_role_selector.py \
  --root "${OUTDIR}" \
  --policy "${POLICY}" \
  --output "${PROPOSAL}" \
  --require-complete

echo "[19F] Docker-only smoke tests"
pytest tests/test_round19_stage19f_*.py -q --tb=short

echo "[19F] Validate all unique role candidates have 15 checkpoints"
python3 tools/round19_stage19f_manifest_builder.py \
  --stage19d-manifest "${MANIFEST_19D}" \
  --proposal-roles "${PROPOSAL}" \
  --output "${INVENTORY}"

python3 - <<PY
import hashlib
import json
from pathlib import Path

import pandas as pd

proposal_path = Path("${PROPOSAL}")
inventory_path = Path("${INVENTORY}")
role_lock = Path("${ROLE_LOCK}")
if role_lock.exists():
    raise AssertionError("Final role lock must not exist at proposal-review gate")

proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
inventory = pd.read_csv(inventory_path)
if proposal.get("proposal_only") is not True:
    raise AssertionError("Proposal must be explicitly proposal_only")
if proposal.get("selection_used_internal") is not False:
    raise AssertionError("Internal benchmark must not participate in role selection")
if proposal.get("selection_used_tcga") is not False:
    raise AssertionError("TCGA benchmark must not participate in role selection")
if proposal.get("single_champion") is not None:
    raise AssertionError("Round 19F does not define a single champion")
for candidate, group in inventory.groupby("source_candidate_id"):
    if len(group) != 15 or group["member_id"].nunique() != 15:
        raise AssertionError(f"{candidate}: incomplete 15-member inventory")
    if not all(Path(path).is_file() for path in group["checkpoint_path"]):
        raise AssertionError(f"{candidate}: checkpoint file missing")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "stage": "19f",
    "gate": "PROPOSAL_REVIEW",
    "smoke_only_requested": bool(${SMOKE_ONLY}),
    "proposal_only": True,
    "final_role_lock_created": False,
    "n_unique_source_candidates": int(inventory["source_candidate_id"].nunique()),
    "n_checkpoints": int(len(inventory)),
    "required_members_per_candidate": 15,
    "proposal_sha256": sha(proposal_path),
    "inventory_sha256": sha(inventory_path),
    "selection_used_internal": False,
    "selection_used_tcga": False,
}
Path("${SUMMARY}").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

r19_notify --event stage-done --stage 19f-proposal \
  --extra "smoke passed; role proposal + 15-member inventory ready for human review"
trap - EXIT
echo "========== ROUND19 STAGE 19F PROPOSAL REVIEW READY =========="
