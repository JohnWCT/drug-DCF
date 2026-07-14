#!/usr/bin/env bash
# Round 19B training pilot (6 full train_fold jobs; NOT formal dispatcher)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
PILOT_ROOT="${OUTDIR}/stage19b_pilot"

echo "========== ROUND19 STAGE 19B PILOT START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

python - <<'PY'
import json
from pathlib import Path
settings = json.loads(Path("config/round19_factorial_settings.json").read_text())
assert settings["stage19b_expected_jobs"] == 117
assert settings["stage19b_omics_anchors"] == ["O1", "O2", "O3"]
cells = settings["stage19b_pilot_cells"]
assert len(cells) == 6
print("pilot cells:")
for c in cells:
    print(" ", c)
PY

# Ensure 117-job manifest exists
python tools/round19_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 19b_manifest
python - <<'PY'
import pandas as pd
df = pd.read_csv("result/optimization_runs/round19_factorial/manifests/stage19b_drug_predictor_manifest.csv")
assert len(df) == 117, len(df)
assert set(df.omics_id) == {"O1", "O2", "O3"}
assert len(df[(df.drug_representation_id=="D1")&(df.predictor_id=="P2")]) == 0
assert len(df[(df.drug_representation_id=="D4")&(df.predictor_id=="P2")]) == 0
print("manifest 117 OK")
PY

FOLD=0
MAX_EPOCHS="${PILOT_MAX_EPOCHS:-80}"
MICRO="${PILOT_MICRO_BATCH:-64}"
ACCUM="${PILOT_ACCUM:-16}"

python - <<'PY' > /tmp/round19_pilot_jobs.txt
import json
from pathlib import Path
settings = json.loads(Path("config/round19_factorial_settings.json").read_text())
for c in settings["stage19b_pilot_cells"]:
    print(f"{c['drug']} {c['predictor']} {c['omics']} {c['purpose']}")
PY

while read -r DRUG PRED OMICS PURPOSE; do
  RDIR="${PILOT_ROOT}/${DRUG}_${PRED}_${OMICS}_fold${FOLD}"
  mkdir -p "${RDIR}"
  echo "[pilot] ${DRUG}×${PRED}×${OMICS} (${PURPOSE}) -> ${RDIR}"
  python step1_finetune_latent_pipeline_round19.py \
    --mode train_fold \
    --pilot \
    --settings "${SETTINGS}" \
    --result-dir "${RDIR}" \
    --response-path "${OUTDIR}/data/round19_eligible_response.csv" \
    --split-assignment "${OUTDIR}/splits/screening_3fold_assignments.csv" \
    --internal-test-path "${OUTDIR}/splits/internal_test_split.csv" \
    --drug-id "${DRUG}" \
    --predictor-id "${PRED}" \
    --omics-id "${OMICS}" \
    --fold-id "${FOLD}" \
    --model-seed 101 \
    --micro-batch-size "${MICRO}" \
    --accumulation-steps "${ACCUM}" \
    --max-epochs "${MAX_EPOCHS}" \
    --early-stop-patience 20 \
    --early-stop-start-epoch 10
  # artifact assert
  python - <<PY
from pathlib import Path
rd = Path("${RDIR}")
need = [
  "checkpoint.pt","train_history.csv","train_summary.json",
  "val_predictions.csv","val_metrics.json","runtime_resource_summary.json",
  "pilot_job_status.json"
]
missing = [n for n in need if not (rd/n).is_file()]
assert not missing, missing
assert not (rd/"job_status.json").is_file(), "pilot must not write formal job_status.json"
print("artifacts OK", rd)
PY
done < /tmp/round19_pilot_jobs.txt

echo "========== ROUND19 STAGE 19B PILOT DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
