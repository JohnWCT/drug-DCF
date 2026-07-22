#!/usr/bin/env python3
"""Diagnose XA query/drug utilization and attention health (Round 23)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.diagnostics.attention_health import attention_health_summary
from biocda.diagnostics.query_utilization import (
    attention_intervention_suite,
    drug_replacement_sensitivity,
    query_residual_dominance,
    utilization_report,
)
from biocda.models.xa.factory import build_xa_v2
from tools.biocda_telegram_notify import biocda_notify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/biocda/xa_v2_closure.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-type", default="biocda_xa_transfer")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = build_xa_v2(config, model_type=args.model_type)
    if args.checkpoint and args.checkpoint.is_file():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
    model.eval()

    omics = torch.randn(4, 64)
    ctx = torch.randn(4, 32)
    g_a = batch_drug_graphs([make_chain_graph(8, drug_id="a") for _ in range(4)])
    g_b = batch_drug_graphs([make_chain_graph(5, drug_id="b") for _ in range(4)])

    with torch.no_grad():
        full = model(omics, ctx, g_a, output_mode="full")
        residual = query_residual_dominance(full.initial_query, full.final_query)
        # attention health on last layer mean heads → [B,H,N]
        probs = full.attention_probabilities[-1, :, :, 0, :]
        health = attention_health_summary(probs, full.atom_mask)
    drug = drug_replacement_sensitivity(model, omics, ctx, g_a, g_b)
    attn_int = attention_intervention_suite(model, omics, ctx, g_a)
    util = utilization_report(residual=residual, drug=drug, attention=attn_int)

    reports = ROOT / "reports"
    (reports / "round23_attention_health.json").write_text(json.dumps(health, indent=2) + "\n", encoding="utf-8")
    (reports / "round23_query_drug_utilization.json").write_text(json.dumps(util, indent=2) + "\n", encoding="utf-8")

    # C32 effect placeholder — filled after X3 exists
    c32 = {
        "status": "pending_x3",
        "note": "Compare biocda_xa_z_only vs best Z+C strategy after training",
    }
    (reports / "round23_c32_xa_effect.json").write_text(json.dumps(c32, indent=2) + "\n", encoding="utf-8")

    biocda_notify(f"Round23 diagnose util_pass={util['pass']} attn_entropy={health.get('mean_normalized_entropy')}")
    print(json.dumps({"utilization_pass": util["pass"], "attention_health": health}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
