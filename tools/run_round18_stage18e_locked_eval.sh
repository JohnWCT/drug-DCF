#!/usr/bin/env bash
# Round 18 Stage 18E: locked-candidate internal test + TCGA inference + analysis
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=tools/run_round18_notify_helpers.sh
source "$(dirname "$0")/run_round18_notify_helpers.sh"

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
RUN_INTERNAL="${RUN_INTERNAL:-1}"
RUN_TCGA="${RUN_TCGA:-1}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
LIMIT_JOBS="${LIMIT_JOBS:-}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-8}"
ROUND18_NUM_WORKERS="${ROUND18_NUM_WORKERS:-0}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
export ROUND18_NUM_WORKERS
export ROUND18_PIN_MEMORY="${ROUND18_PIN_MEMORY:-1}"
CURRENT_STAGE="18E"
LOCK="${OUTDIR}/reports/round18_locked_selection.json"

trap 'r18_notify --event stage-fail --stage "${CURRENT_STAGE}" --reason "exit code $?"' ERR

echo "========== ROUND18 STAGE 18E START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "SMOKE_ONLY=${SMOKE_ONLY} RUN_INTERNAL=${RUN_INTERNAL} RUN_TCGA=${RUN_TCGA} MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU}"

if [[ ! -f "${LOCK}" ]]; then
  echo "Missing lock file: ${LOCK}" >&2
  exit 1
fi

python tools/round18_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 18e

INTERNAL_MANIFEST="${OUTDIR}/manifests/stage18e_internal_test_manifest.csv"
TCGA_MANIFEST="${OUTDIR}/manifests/stage18e_tcga_manifest.csv"

python - <<'PY'
import json
from pathlib import Path
import pandas as pd

lock = json.loads(Path("result/optimization_runs/round18_architecture/reports/round18_locked_selection.json").read_text())
cands = lock["formal_candidates"]
assert len(cands) == 5, len(cands)
internal = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18e_internal_test_manifest.csv")
tcga = pd.read_csv("result/optimization_runs/round18_architecture/manifests/stage18e_tcga_manifest.csv")
assert len(internal) == 25, len(internal)
assert len(tcga) == 125, len(tcga)
assert set(internal["architecture_id"]) == {c["architecture_id"] for c in cands}
assert set(tcga["mode"]) == {"infer_tcga"}
assert set(internal["mode"]) == {"infer_internal_test"}
# checkpoints must exist for all 18D folds
missing = [p for p in internal["checkpoint_path"] if not Path(p).is_file()]
assert not missing, f"missing checkpoints e.g. {missing[:3]}"
print("18E manifests OK: internal=25 tcga=125 candidates=5")
print(internal.groupby(["architecture_id"]).size().to_string())
PY

r18_notify --event stage-start --stage "${CURRENT_STAGE}" \
  --extra "5 candidates × (25 internal + 125 TCGA) inference; MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU}"

EXTRA=()
if [[ -n "${LIMIT_JOBS}" ]]; then EXTRA+=(--limit "${LIMIT_JOBS}"); fi

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  echo "[18E] SMOKE_ONLY=1: one internal + one TCGA inference job"
  python - <<'PY'
import pandas as pd
from pathlib import Path

def one(src, out, tag):
    df = pd.read_csv(src)
    # Prefer X3 pure for smoke if present
    pref = df[df["architecture_id"].astype(str).str.contains("X3__pure")]
    row = (pref if len(pref) else df).iloc[0].to_dict()
    row["job_id"] = str(row["job_id"]) + f"_{tag}_smoke"
    row["result_dir"] = str(row["result_dir"]).rstrip("/") + "_smoke"
    row["requested_micro_batch"] = 64
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print("wrote", out, row["architecture_id"], row.get("target_key"))

one(
    "result/optimization_runs/round18_architecture/manifests/stage18e_internal_test_manifest.csv",
    "result/optimization_runs/round18_architecture/manifests/stage18e_smoke_internal.csv",
    "internal",
)
one(
    "result/optimization_runs/round18_architecture/manifests/stage18e_tcga_manifest.csv",
    "result/optimization_runs/round18_architecture/manifests/stage18e_smoke_tcga.csv",
    "tcga",
)
PY
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18e_smoke_internal.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 \
    --micro-batch-candidates 64,32 \
    --limit 1
  python tools/round18_oom_runner.py dispatch \
    --manifest "${OUTDIR}/manifests/stage18e_smoke_tcga.csv" \
    --pipeline step1_finetune_latent_pipeline_round18_cv.py \
    --max-jobs-per-gpu 1 \
    --micro-batch-candidates 64,32 \
    --limit 1
else
  if [[ "${RUN_INTERNAL}" == "1" ]]; then
    echo "[18E] dispatching internal-test inference (25 jobs)"
    python tools/round18_oom_runner.py dispatch \
      --manifest "${INTERNAL_MANIFEST}" \
      --pipeline step1_finetune_latent_pipeline_round18_cv.py \
      --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" \
      --micro-batch-candidates 512,256,128,64,32 \
      "${EXTRA[@]}"
  fi
  if [[ "${RUN_TCGA}" == "1" ]]; then
    echo "[18E] dispatching TCGA inference (125 jobs)"
    python tools/round18_oom_runner.py dispatch \
      --manifest "${TCGA_MANIFEST}" \
      --pipeline step1_finetune_latent_pipeline_round18_cv.py \
      --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" \
      --micro-batch-candidates 512,256,128,64,32 \
      "${EXTRA[@]}"
  fi
  if [[ "${RUN_ANALYZE}" == "1" ]]; then
    echo "[18E] analyzing ensembles + paired bootstrap"
    python tools/analyze_round18_external_eval.py \
      --outdir "${OUTDIR}" \
      --n-bootstrap "${N_BOOTSTRAP}"
  fi
fi

r18_notify --event stage-done --stage "${CURRENT_STAGE}" --manifest "${INTERNAL_MANIFEST}"
echo "========== ROUND18 STAGE 18E DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
