#!/usr/bin/env bash
# Round 19 Stage 19A: setup + smoke tests
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SETTINGS="${ROUND19_SETTINGS:-config/round19_factorial_settings.json}"
OUTDIR="${ROUND19_ROOT:-result/optimization_runs/round19_factorial}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" "${OUTDIR}"

echo "========== ROUND19 STAGE 19A SETUP SMOKE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[19A] py_compile"
python -m py_compile \
  drugmodels/ginconv.py \
  drugmodels/gineconv.py \
  tools/round19_graph_features.py \
  tools/round19_drug_features.py \
  tools/round19_drug_encoders.py \
  tools/round19_fusion_models.py \
  tools/round19_feature_builder.py \
  tools/round19_config_builder.py

echo "[19A] pytest round19 unit tests"
pytest tests/test_round19_*.py -q

echo "[19A] build features / cache / git baseline"
python tools/round19_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 19a

echo "[19A] five drug-family live smokes (D0+P0, D1+P1, D2+P2, D3+P2, D4+P1)"
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
df = pd.read_csv(drug_csv)
# Prefer eligible-like rows with valid SMILES
df = df.dropna(subset=["SMILES", "DRUG_NAME"]).drop_duplicates("DRUG_NAME")
smiles_rows = df.head(8)
smiles_list = smiles_rows["SMILES"].astype(str).tolist()
drug_names = smiles_rows["DRUG_NAME"].astype(str).tolist()

maccs_map = load_maccs_by_drug_name(drug_csv, drug_names=drug_names)
validate_maccs_coverage(maccs_map, drug_names)

cells = settings["stage19a_smoke_cells"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for cell in cells:
    drug_id = cell["drug"]
    pred_id = cell["predictor"]
    omics_id = cell["omics"]
    assert_compatible(drug_id, pred_id)
    drug_cfg = settings["drug_representations"][drug_id]
    omics_dim = resolve_omics_dim(omics_id)
    enc_type = drug_cfg["type"]

    print(f"  smoke {drug_id}+{pred_id}+{omics_id} ...", flush=True)
    B = 3
    omics = torch.randn(B, omics_dim, device=device)

    if enc_type == "maccs":
        assert_no_hybrid(enc_type, has_maccs=True, has_graph=False)
        encoder = build_drug_encoder("maccs", maccs_output_dim=int(drug_cfg["output_dim"])).to(device)
        encoder.train()
        x = torch.tensor(
            [maccs_map[d] for d in drug_names[:B]], dtype=torch.float32, device=device
        )
        drug_vec = encoder(x)
        assert drug_vec.shape == (B, 64)
        fusion, head = build_predictor_and_head(pred_id, omics_dim=omics_dim, drug_dim=64)
        fusion = fusion.to(device)
        head = head.to(device)
        repr_vec = fusion(omics, drug_vec)
        logits = head(repr_vec)
        loss = logits.float().pow(2).mean()
        loss.backward()
        grads = [p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None]
        assert grads and max(grads) > 0, "MACCS encoder grads missing"
        assert torch.isfinite(logits).all()
        print(f"    OK MACCS logits={tuple(logits.shape)} loss={float(loss):.4f}")
        continue

    with_bonds = bool(drug_cfg.get("edge_features"))
    assert_no_hybrid(enc_type, has_maccs=False, has_graph=True)
    graphs = [build_pyg_data(s, with_bonds=with_bonds) for s in smiles_list[:B]]
    batch = Batch.from_data_list(graphs).to(device)
    encoder = build_drug_encoder(
        enc_type,
        node_hidden_dim=int(drug_cfg["node_hidden_dim"]),
        graph_output_dim=int(drug_cfg["graph_output_dim"]),
        edge_dim=int(BOND_FEATURE_DIM if drug_cfg.get("edge_features") else drug_cfg.get("edge_dim", BOND_FEATURE_DIM)),
        dropout=0.1,
    ).to(device)
    encoder.train()
    bn_before = None
    if hasattr(encoder, "bns") and encoder.bns and encoder.bns[0] is not None:
        bn_before = encoder.bns[0].running_mean.detach().clone()

    if pred_id == "P2":
        out = encoder(batch, return_dict=True, return_graph_embedding=False)
        node_emb = out["node_embeddings"]
        assert node_emb.shape[-1] == int(drug_cfg["node_hidden_dim"]) or (
            # JK last -> node_hidden_dim
            node_emb.shape[-1] == encoder.node_dim
        )
        fusion, head = build_predictor_and_head(
            pred_id, omics_dim=omics_dim, drug_dim=0, node_dim=int(node_emb.shape[-1])
        )
        fusion = fusion.to(device)
        head = head.to(device)
        repr_vec = fusion(omics, node_emb, out["batch_index"])
    else:
        out = encoder(batch, return_dict=True)
        drug_vec = out["graph_embedding"]
        assert drug_vec.shape == (B, int(drug_cfg["graph_output_dim"]))
        fusion, head = build_predictor_and_head(
            pred_id, omics_dim=omics_dim, drug_dim=int(drug_vec.shape[-1])
        )
        fusion = fusion.to(device)
        head = head.to(device)
        repr_vec = fusion(omics, drug_vec)
        if bn_before is not None:
            after = encoder.bns[0].running_mean.detach()
            assert not torch.allclose(bn_before, after), "BatchNorm running stats did not update"

    logits = head(repr_vec)
    loss = logits.float().pow(2).mean()
    loss.backward()
    enc_grads = [p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None]
    assert enc_grads and max(enc_grads) > 0, f"{enc_type} encoder grads missing"
    if enc_type == "gine":
        # edge encoder lives inside GINEConv.nn / edge MLP parameters
        edge_grad = 0.0
        for name, p in encoder.named_parameters():
            if p.grad is None:
                continue
            if "convs" in name:
                edge_grad += float(p.grad.abs().sum().item())
        assert edge_grad > 0, "GINE conv grads missing"
    assert torch.isfinite(logits).all()
    print(f"    OK {enc_type} logits={tuple(logits.shape)} loss={float(loss):.4f}")

print("All five family smokes passed")
PY

echo "========== ROUND19 STAGE 19A SETUP SMOKE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
