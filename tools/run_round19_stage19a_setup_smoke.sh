#!/usr/bin/env bash
# Round 19 Stage 19A+: full infrastructure smoke (Docker DAPL)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "========== ROUND19 FULL INFRA SMOKE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[19 smoke] py_compile"
python -m py_compile \
  drugmodels/ginconv.py \
  drugmodels/gineconv.py \
  tools/round19_graph_features.py \
  tools/round19_drug_features.py \
  tools/round19_drug_encoders.py \
  tools/round19_fusion_models.py \
  tools/round19_feature_builder.py \
  tools/round19_config_builder.py \
  tools/round19_dataset.py \
  tools/round19_cv_splits.py \
  tools/round19_scaffold_groups.py \
  tools/round19_manifest_validator.py \
  tools/round19_selection_lock.py \
  tools/round19_oom_runner.py \
  tools/analyze_round19.py \
  step1_finetune_latent_pipeline_round19.py

echo "[19 smoke] pytest round19"
pytest tests/test_round19_*.py -q

echo "[19 smoke] stage 19a setup (features/cache/eligible/splits/19b manifest)"
python tools/round19_config_builder.py --settings "${SETTINGS}" --outdir "${OUTDIR}" --stage 19a

echo "[19 smoke] analyzer template + lock guard"
python tools/analyze_round19.py --stage 19b --outdir "${OUTDIR}"
python tools/analyze_round19.py --stage selection --write-lock --outdir "${OUTDIR}" && exit 1 || true

echo "[19 smoke] five-family live tensor smoke"
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Batch
from tools.round19_drug_encoders import assert_no_hybrid, build_drug_encoder
from tools.round19_drug_features import load_maccs_by_drug_name, validate_maccs_coverage
from tools.round19_feature_builder import resolve_omics_dim
from tools.round19_fusion_models import assert_compatible, build_predictor_and_head
from tools.round19_graph_features import BOND_FEATURE_DIM, build_pyg_data

settings = json.loads(Path("config/round19_factorial_settings.json").read_text())
drug_csv = settings["drug_smiles_path"]
df = pd.read_csv(drug_csv).dropna(subset=["SMILES", "DRUG_NAME"]).drop_duplicates("DRUG_NAME").head(8)
smiles_list = df["SMILES"].astype(str).tolist()
drug_names = df["DRUG_NAME"].astype(str).tolist()
maccs_map = load_maccs_by_drug_name(drug_csv, drug_names=drug_names)
validate_maccs_coverage(maccs_map, drug_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for cell in settings["stage19a_smoke_cells"]:
    drug_id, pred_id, omics_id = cell["drug"], cell["predictor"], cell["omics"]
    assert_compatible(drug_id, pred_id)
    drug_cfg = settings["drug_reps"][drug_id]
    omics_dim = resolve_omics_dim(omics_id)
    enc_type = drug_cfg["type"]
    B = 3
    omics = torch.randn(B, omics_dim, device=device)
    print(f"  tensor smoke {drug_id}+{pred_id}+{omics_id}", flush=True)
    if enc_type == "maccs":
        assert_no_hybrid(enc_type, has_maccs=True, has_graph=False)
        encoder = build_drug_encoder("maccs", maccs_output_dim=int(drug_cfg["output_dim"])).to(device)
        x = torch.tensor([maccs_map[d] for d in drug_names[:B]], dtype=torch.float32, device=device)
        drug_vec = encoder(x)
        fusion, head = build_predictor_and_head(pred_id, omics_dim=omics_dim, drug_dim=64)
        fusion, head = fusion.to(device), head.to(device)
        logits = head(fusion(omics, drug_vec))
        loss = logits.float().pow(2).mean(); loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        assert torch.isfinite(logits).all()
        continue
    assert_no_hybrid(enc_type, has_maccs=False, has_graph=True)
    with_bonds = bool(drug_cfg.get("edge_features"))
    batch = Batch.from_data_list([build_pyg_data(s, with_bonds=with_bonds) for s in smiles_list[:B]]).to(device)
    encoder = build_drug_encoder(
        enc_type,
        node_hidden_dim=int(drug_cfg["node_hidden_dim"]),
        graph_output_dim=int(drug_cfg["graph_output_dim"]),
        edge_dim=int(BOND_FEATURE_DIM if with_bonds else drug_cfg.get("edge_dim", BOND_FEATURE_DIM)),
    ).to(device)
    encoder.train()
    bn_before = encoder.bns[0].running_mean.detach().clone() if encoder.bns and encoder.bns[0] is not None else None
    if pred_id == "P2":
        out = encoder(batch, return_dict=True, return_graph_embedding=False)
        fusion, head = build_predictor_and_head(pred_id, omics_dim=omics_dim, drug_dim=0, node_dim=int(out["node_embeddings"].shape[-1]))
        fusion, head = fusion.to(device), head.to(device)
        repr_vec = fusion(omics, out["node_embeddings"], out["batch_index"])
    else:
        out = encoder(batch, return_dict=True)
        fusion, head = build_predictor_and_head(pred_id, omics_dim=omics_dim, drug_dim=int(out["graph_embedding"].shape[-1]))
        fusion, head = fusion.to(device), head.to(device)
        repr_vec = fusion(omics, out["graph_embedding"])
        if bn_before is not None:
            assert not torch.allclose(bn_before, encoder.bns[0].running_mean.detach())
    logits = head(repr_vec)
    loss = logits.float().pow(2).mean(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert torch.isfinite(logits).all()
print("tensor family smokes OK")
PY

echo "[19 smoke] real-data pipeline smoke for 5 cells"
for cell in \
  "D0 P0 O1" \
  "D1 P1 O3" \
  "D2 P2 O3" \
  "D3 P2 O1" \
  "D4 P1 O3"
do
  set -- ${cell}
  echo "  data_smoke $1+$2+$3"
  python step1_finetune_latent_pipeline_round19.py \
    --mode data_smoke \
    --settings "${SETTINGS}" \
    --outdir "${OUTDIR}/data_smoke/${1}_${2}_${3}" \
    --response-path "${OUTDIR}/data/round19_eligible_response.csv" \
    --split-assignment "${OUTDIR}/splits/screening_3fold_assignments.csv" \
    --drug-id "$1" \
    --predictor-id "$2" \
    --omics-id "$3" \
    --fold-id 0 \
    --micro-batch-size 8 \
    --max-batches 2 \
    --max-rows 48
done

echo "[19 smoke] scaffold + heldout split smoke"
python - <<'PY'
import pandas as pd
from tools.round18_eligible_data import load_smiles_lookup
from tools.round19_cv_splits import attach_scaffold_ids, build_heldout_assignments
from tools.round19_scaffold_groups import murcko_scaffold_id

eligible = pd.read_csv("result/optimization_runs/round19_factorial/data/round19_eligible_response.csv")
# tiny subset for smoke speed
sub = eligible.drop_duplicates("DRUG_NAME").head(30).copy()
smiles = load_smiles_lookup("data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv")
# map by lower key
drug_to_smiles = {}
for d in sub["DRUG_NAME"].astype(str):
    key = d.strip().lower()
    if key not in smiles:
        raise KeyError(d)
    drug_to_smiles[d] = smiles[key]
sub = attach_scaffold_ids(sub, drug_to_smiles)
assert "scaffold_id" in sub.columns
assert sub["scaffold_id"].notna().all()
# drug-held-out on small development-like frame needs enough groups; use ModelID on larger sample
dev = eligible.head(2000).copy()
asg = build_heldout_assignments(dev, group_column="DRUG_NAME", n_splits=5, split_seed=52, cv_name="drug_heldout_5fold_smoke")
assert set(asg["fold_id"]) == set(range(5))
print("scaffold/heldout smoke OK", murcko_scaffold_id("c1ccccc1"), "n_assign", len(asg))
PY

echo "========== ROUND19 FULL INFRA SMOKE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
