#!/usr/bin/env python3
"""Round 19 pipeline entry (smoke / future train_fold)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_dataset import subset_by_assignment
from tools.round18_response_head import Round18ResponseHead
from tools.round19_drug_encoders import assert_no_hybrid, build_drug_encoder
from tools.round19_feature_builder import OMICS_ALIAS, resolve_omics_dim
from tools.round19_fusion_models import assert_compatible, build_predictor
from tools.round19_graph_features import BOND_FEATURE_DIM
from tools.round19_dataset import Round19ResponseDataset, round19_collate_fn


def _load_settings(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _feature_dir(settings: dict, omics_id: str) -> str:
    root = settings.get(
        "round19_feature_out_root", "result/optimization_runs/round19_factorial/features"
    )
    return str(Path(root) / OMICS_ALIAS[omics_id])


def run_data_smoke(args: argparse.Namespace) -> dict:
    settings = _load_settings(args.settings)
    outdir = Path(args.outdir)
    eligible = pd.read_csv(args.response_path)
    assignments = pd.read_csv(args.split_assignment)
    train_df = subset_by_assignment(eligible, assignments, fold_id=args.fold_id, split_role="train")
    # keep smoke fast
    train_df = train_df.head(int(args.max_rows)).copy()

    drug_cfg = settings["drug_reps"][args.drug_id]
    enc_type = drug_cfg["type"]
    with_bonds = bool(drug_cfg.get("edge_features"))
    assert_compatible(args.drug_id, args.predictor_id)
    assert_no_hybrid(
        enc_type,
        has_maccs=(enc_type == "maccs"),
        has_graph=(enc_type in {"gin", "gine"}),
    )

    ds = Round19ResponseDataset(
        train_df,
        feature_dir=_feature_dir(settings, args.omics_id),
        drug_smiles_path=settings["drug_smiles_path"],
        encoder_type=enc_type,
        with_bonds=with_bonds,
    )
    assert ds.omics_dim == resolve_omics_dim(args.omics_id)
    loader = DataLoader(
        ds,
        batch_size=int(args.micro_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=round19_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if enc_type == "maccs":
        encoder = build_drug_encoder("maccs", maccs_output_dim=int(drug_cfg["output_dim"])).to(device)
        drug_dim = int(drug_cfg["output_dim"])
        node_dim = 32
    else:
        encoder = build_drug_encoder(
            enc_type,
            node_hidden_dim=int(drug_cfg["node_hidden_dim"]),
            graph_output_dim=int(drug_cfg["graph_output_dim"]),
            edge_dim=int(drug_cfg.get("edge_dim", BOND_FEATURE_DIM)),
        ).to(device)
        drug_dim = int(drug_cfg["graph_output_dim"])
        node_dim = int(drug_cfg["node_hidden_dim"])

    fusion = build_predictor(
        args.predictor_id, omics_dim=ds.omics_dim, drug_dim=drug_dim, node_dim=node_dim
    ).to(device)
    head = Round18ResponseHead(input_dim=fusion.output_dim).to(device)
    encoder.train()
    fusion.train()
    head.train()

    n_batches = 0
    last_loss = None
    for batch in loader:
        if n_batches >= int(args.max_batches):
            break
        omics = batch["omics"].to(device)
        if enc_type == "maccs":
            drug_vec = encoder(batch["maccs"].to(device))
            if args.predictor_id == "P2":
                raise AssertionError("P2 incompatible with MACCS")
            repr_vec = fusion(omics, drug_vec)
        else:
            drug_batch = batch["drug_batch"].to(device)
            if args.predictor_id == "P2":
                out = encoder(drug_batch, return_dict=True, return_graph_embedding=False)
                repr_vec = fusion(omics, out["node_embeddings"], out["batch_index"])
            else:
                out = encoder(drug_batch, return_dict=True)
                repr_vec = fusion(omics, out["graph_embedding"])
        logits = head(repr_vec)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), batch["label"].to(device)
        )
        loss.backward()
        last_loss = float(loss.detach().cpu())
        assert torch.isfinite(logits).all()
        n_batches += 1

    summary = {
        "mode": "data_smoke",
        "drug_id": args.drug_id,
        "predictor_id": args.predictor_id,
        "omics_id": args.omics_id,
        "n_rows": int(len(ds)),
        "n_batches": int(n_batches),
        "omics_dim": int(ds.omics_dim),
        "encoder_type": enc_type,
        "last_loss": last_loss,
        "device": str(device),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 pipeline")
    parser.add_argument("--mode", default="data_smoke", choices=["data_smoke"])
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round19_factorial/data_smoke")
    parser.add_argument(
        "--response-path",
        default="result/optimization_runs/round19_factorial/data/round19_eligible_response.csv",
    )
    parser.add_argument(
        "--split-assignment",
        default="result/optimization_runs/round19_factorial/splits/screening_3fold_assignments.csv",
    )
    parser.add_argument("--drug-id", default="D0")
    parser.add_argument("--predictor-id", default="P0")
    parser.add_argument("--omics-id", default="O1")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "data_smoke":
        run_data_smoke(args)
    else:
        raise SystemExit(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
