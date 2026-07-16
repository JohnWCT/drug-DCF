#!/usr/bin/env bash
# Round 19H reproducibility/archive planning. Never cleans, copies checkpoints,
# changes links, modifies the final role lock, commits, or contacts Git remotes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

ROUND19_ROOT="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
REPORTS="${ROUND19_ROOT}/reports"
MANIFESTS="${ROUND19_ROOT}/manifests"
FINAL_LOCK="${ROUND19_FINAL_ROLE_LOCK:-${REPORTS}/round19_final_role_lock.json}"
OUTDIR="${ROUND19_STAGE19H_OUTDIR:-${ROUND19_ROOT}/stage19h_reproducibility}"
REPORT_19G="${ROUND19_STAGE19G_REPORT:-}"
STAGE19G_OUTPUT="${ROUND19_STAGE19G_OUTPUT:-${ROUND19_ROOT}/stage19g}"
REPOSITORY_ATTESTATION="${ROUND19_REPOSITORY_ATTESTATION:-}"

DRY_RUN=1
REQUIRE_COMPLETE=1
for arg in "$@"; do
  case "${arg}" in
    --write-plan) DRY_RUN=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --allow-incomplete) REQUIRE_COMPLETE=0 ;;
    --report-19g=*) REPORT_19G="${arg#*=}" ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

COMMON=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  COMMON+=(--dry-run)
fi
INCOMPLETE=()
if [[ "${REQUIRE_COMPLETE}" -eq 0 ]]; then
  INCOMPLETE+=(--allow-incomplete)
fi
ATTESTATION_ARGS=()
if [[ -n "${REPOSITORY_ATTESTATION}" ]]; then
  ATTESTATION_ARGS+=(--repository-attestation "${REPOSITORY_ATTESTATION}")
fi

if [[ "${REQUIRE_COMPLETE}" -eq 1 ]]; then
  [[ -f "${FINAL_LOCK}" ]] || {
    echo "19H prerequisite failed: missing final role lock: ${FINAL_LOCK}" >&2
    exit 3
  }
  python3 - "${FINAL_LOCK}" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
if len(inventory) != 90:
    raise SystemExit(f"19H prerequisite failed: expected 90 locked checkpoints, got {len(inventory)}")
PY
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${OUTDIR}"
fi

python3 tools/round19_reproducibility_audit.py \
  --project-root "${ROOT}" \
  --final-lock "${FINAL_LOCK}" \
  --include "${REPORTS}" \
  --include "${MANIFESTS}" \
  --include "${STAGE19G_OUTPUT}" \
  "${ATTESTATION_ARGS[@]}" \
  --output "${OUTDIR}/reproducibility_audit.json" \
  "${COMMON[@]}" "${INCOMPLETE[@]}"

python3 tools/round19_artifact_manifest.py \
  --project-root "${ROOT}" \
  --final-lock "${FINAL_LOCK}" \
  --inventory-root "${REPORTS}" \
  --inventory-root "${MANIFESTS}" \
  --inventory-root "${STAGE19G_OUTPUT}" \
  --manifest-seed "${FINAL_LOCK}" \
  --output "${OUTDIR}/artifact_manifest.json" \
  --portable-mapping-output "${OUTDIR}/portable_symlink_mapping.json" \
  "${COMMON[@]}" "${INCOMPLETE[@]}"

MODEL_ARGS=(
  --final-lock "${FINAL_LOCK}"
  --output "${OUTDIR}/model_card.json"
)
DATASET_ARGS=(
  --project-root "${ROOT}"
  --dataset "gdsc_intersect13=data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv"
  --dataset "tcga_only3=data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv"
  --dataset "dapl=data/TCGA/TCGA_drug_response_from_DAPL.csv"
  --dataset "aacdr_tcga_only=data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv"
  --dataset "aacdr_gdsc_intersect=data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv"
  --output "${OUTDIR}/dataset_card.json"
)
if [[ -n "${REPORT_19G}" ]]; then
  MODEL_ARGS+=(--report-19g "${REPORT_19G}")
  DATASET_ARGS+=(--report-19g "${REPORT_19G}")
fi

python3 tools/round19_model_card_builder.py "${MODEL_ARGS[@]}" "${COMMON[@]}"
python3 tools/round19_dataset_card_builder.py "${DATASET_ARGS[@]}" "${COMMON[@]}"

echo "Round 19H plan generated (dry_run=${DRY_RUN}, require_complete=${REQUIRE_COMPLETE})."
echo "Stage 19G status: $([[ -n "${REPORT_19G}" ]] && echo supplied || echo awaiting_19g)."
echo "No cleanup, checkpoint copy, symlink rewrite, role-lock change, Git remote, commit, or ALL_DONE marker occurred."
