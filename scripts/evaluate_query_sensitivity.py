#!/usr/bin/env python3
"""Query sensitivity diagnostics CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.diagnostics.query_sensitivity import compare_attention
from biocda.models.model_factory import build_model, build_model_config_for_type
from biocda.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/biocda/xa_validation.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/query_sensitivity_summary.csv",
    )
    args = parser.parse_args()

    base = yaml.safe_load((ROOT / "configs/model/biocda_cross_attention.yaml").read_text())
    cfg = build_model_config_for_type(base, "biocda_xa_zc")
    set_seed(17)
    model = build_model(cfg)
    model.eval()

    omics = torch.randn(4, 64)
    context = torch.randn(4, 32)
    batch = batch_drug_graphs([make_chain_graph(8)] * 4)

    with torch.no_grad():
        base_out = model(omics, context, batch, output_mode="attention")
        shuffled_omics = omics[torch.randperm(4)]
        shuf_out = model(shuffled_omics, context, batch, output_mode="attention")

    same_drug = compare_attention(
        base_out.atom_attention[:2], base_out.atom_attention[2:], base_out.atom_mask[:2]
    )
    query_shuffle = compare_attention(
        base_out.atom_attention, shuf_out.atom_attention, base_out.atom_mask
    )
    row = {**same_drug, **{f"shuffle_{k}": v for k, v in query_shuffle.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
