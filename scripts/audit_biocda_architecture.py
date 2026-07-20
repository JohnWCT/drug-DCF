#!/usr/bin/env python3
"""Runtime BioCDA architecture contract audit with forward hooks."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.model_factory import build_model
from biocda.utils.reproducibility import set_seed

TRACE: List[Dict[str, Any]] = []


def _hook(name: str):
    def fn(module, inputs, output):
        inp_shapes = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                inp_shapes.append(list(x.shape))
        out_shape = list(output.shape) if isinstance(output, torch.Tensor) else str(type(output))
        TRACE.append(
            {
                "module": name,
                "input_shapes": inp_shapes,
                "output_shape": out_shape,
                "requires_grad": any(
                    isinstance(x, torch.Tensor) and x.requires_grad for x in inputs
                ),
                "training": module.training,
            }
        )

    return fn


def audit_architecture(config: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
    global TRACE
    TRACE = []
    set_seed(17)
    model = build_model(config)
    model.eval()

    handles = []
    for name, module in model.named_modules():
        if name and any(
            key in name
            for key in ("omics_encoder", "sample_encoder", "drug_encoder", "cross_attention", "response_head")
        ):
            handles.append(module.register_forward_hook(_hook(name)))

    graphs = [make_chain_graph(8), make_chain_graph(3)]
    batch = batch_drug_graphs(graphs)
    omics = torch.randn(2, config["model"]["omics_encoder"]["latent_dim"])
    context = torch.randn(2, config["model"]["biological_context"]["context_dim"])

    with torch.no_grad():
        pred = model(omics, context, batch, output_mode="prediction")
        attn = model(omics, context, batch, output_mode="attention")
        full = model(omics, context, batch, output_mode="full")

    for h in handles:
        h.remove()

    checks: Dict[str, bool] = {}
    checks["sample_representation_in_full"] = full.sample_representation is not None
    checks["omics_latent_in_full"] = full.omics_latent is not None
    checks["biological_context_in_full"] = full.biological_context is not None
    checks["node_embeddings_in_full"] = full.node_embeddings is not None
    checks["atom_attention_exported"] = attn.atom_attention is not None
    checks["attention_logits_exported"] = attn.atom_attention_logits is not None
    checks["atom_mask_exported"] = attn.atom_mask is not None
    checks["atom_ptr_exported"] = attn.atom_ptr is not None
    checks["logits_shape_batch"] = list(pred.logits.shape) == [2]
    checks["output_modes_same_logits"] = torch.allclose(pred.logits, attn.logits) and torch.allclose(
        pred.logits, full.logits
    )

    if attn.atom_attention is not None and attn.atom_mask is not None:
        valid = attn.atom_attention * attn.atom_mask.unsqueeze(1)
        sums = valid.sum(dim=-1)
        checks["attention_sums_to_one"] = bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        pad = attn.atom_attention.masked_select(~attn.atom_mask.unsqueeze(1))
        checks["padding_attention_zero"] = bool((pad == 0).all())
    else:
        checks["attention_sums_to_one"] = False
        checks["padding_attention_zero"] = False

    checks["no_pooled_bypass"] = not hasattr(model, "fusion") or getattr(model, "fusion", None) is None
    checks["cross_attention_present"] = hasattr(model, "cross_attention")

    issues = [k for k, v in checks.items() if not v]
    status = "PASS" if not issues else "FAIL"
    return {
        "status": status,
        "architecture_version": getattr(model, "ARCHITECTURE_VERSION", "unknown"),
        "architecture_name": getattr(model, "architecture_name", config["model"]["type"]),
        "checks": checks,
        "issues": issues,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/biocda/xa_validation.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/biocda_architecture_runtime_audit.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=ROOT / "reports/biocda_forward_trace.json",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    xa_cfg = config["models"][2] if "models" in config else config
    if isinstance(xa_cfg, str):
        model_cfg_path = ROOT / "configs/model/biocda_cross_attention.yaml"
        model_config = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))
        model_config["model"]["type"] = "biocda_xa_zc"
    else:
        from biocda.models.model_factory import build_model_config_for_type

        base = yaml.safe_load((ROOT / "configs/model/biocda_cross_attention.yaml").read_text())
        model_config = build_model_config_for_type(base, "biocda_xa_zc")

    report = audit_architecture(model_config, strict=args.strict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.trace_output.write_text(json.dumps(TRACE, indent=2) + "\n", encoding="utf-8")
    print(f"ARCHITECTURE_RUNTIME_AUDIT={report['status']}")
    if report["status"] == "FAIL" and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
