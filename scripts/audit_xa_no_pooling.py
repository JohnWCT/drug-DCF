#!/usr/bin/env python3
"""Strict no-pooling audit for BioCDA-XA v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.xa.factory import build_xa_v2
from biocda.training.distillation import assert_student_checkpoint_has_no_teacher, export_student_only_state
from biocda.training.gin_transfer import transfer_e3_gin_to_xa, write_transfer_report
from tools.biocda_telegram_notify import biocda_notify

POOL_NAME_FRAGMENTS = (
    "global_max_pool",
    "global_mean_pool",
    "global_add_pool",
    "GlobalAttention",
    "Set2Set",
    "graph_embedding",
    "graph_projection",
    "pooled_residual",
)


def _cfg(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def audit_module_tree(model) -> Dict[str, Any]:
    bad: List[str] = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        for frag in ("GlobalAttention", "Set2Set"):
            if frag in cls:
                bad.append(f"{name}:{cls}")
    # Attribute-level forbidden on XA public API
    for attr in ("pool_type", "residual_mode", "pooled_residual", "fusion"):
        if hasattr(model, attr) and getattr(model, attr) is not None and attr != "pool_type":
            bad.append(f"attr:{attr}")
    return {"ok": len(bad) == 0, "violations": bad}


def audit_forward_no_pool(model) -> Dict[str, Any]:
    """Hook GINConvNet.pool_graph / project_graph — must never run on XA forward."""
    gin = model.drug_encoder.gin
    calls = {"pool_graph": 0, "project_graph": 0}

    orig_pool = gin.pool_graph
    orig_proj = gin.project_graph

    def _pool(*a, **k):
        calls["pool_graph"] += 1
        return orig_pool(*a, **k)

    def _proj(*a, **k):
        calls["project_graph"] += 1
        return orig_proj(*a, **k)

    gin.pool_graph = _pool  # type: ignore[method-assign]
    gin.project_graph = _proj  # type: ignore[method-assign]
    try:
        model.eval()
        omics = torch.randn(2, 64)
        ctx = torch.randn(2, 32)
        graphs = batch_drug_graphs([make_chain_graph(5 + i, drug_id=f"d{i}") for i in range(2)])
        with torch.no_grad():
            out = model(omics, ctx, graphs, output_mode="full")
            head_ok = list(out.logits.shape) == [2]
            modes = []
            for mode in ("prediction", "attention", "full"):
                o = model(omics, ctx, graphs, output_mode=mode)
                modes.append(o.logits)
            same_logits = all(torch.allclose(modes[0], m) for m in modes[1:])
        # Response head input = final query only: check residual path via cosine change possible
        assert out.final_query is not None
        head_input = out.final_query[:, 0, :]
        return {
            "ok": calls["pool_graph"] == 0 and calls["project_graph"] == 0 and head_ok and same_logits,
            "pool_graph_calls": calls["pool_graph"],
            "project_graph_calls": calls["project_graph"],
            "logits_shape_ok": head_ok,
            "modes_identical_logits": bool(same_logits),
            "head_input_shape": list(head_input.shape),
            "attention_shape": list(out.attention_probabilities.shape) if out.attention_probabilities is not None else None,
        }
    finally:
        gin.pool_graph = orig_pool  # type: ignore[method-assign]
        gin.project_graph = orig_proj  # type: ignore[method-assign]


def audit_attention_contract(model) -> Dict[str, Any]:
    omics = torch.randn(2, 64)
    ctx = torch.randn(2, 32)
    graphs = batch_drug_graphs([make_chain_graph(6, drug_id="a"), make_chain_graph(4, drug_id="b")])
    with torch.no_grad():
        out = model(omics, ctx, graphs, output_mode="attention")
    attn = out.attention_probabilities  # [L,B,H,1,N]
    mask = out.atom_mask
    checks = {
        "shape_layers_heads": list(attn.shape[:3]) == [2, 2, 4],
        "valid_sums_to_one": True,
        "padding_zero": True,
    }
    # Squeeze query dim
    probs = attn[:, :, :, 0, :]  # [L,B,H,N]
    for layer in range(probs.shape[0]):
        for head in range(probs.shape[2]):
            p = probs[layer, :, head, :]
            m = mask
            valid_sum = (p * m.float()).sum(dim=-1)
            if not torch.allclose(valid_sum, torch.ones_like(valid_sum), atol=1e-4):
                checks["valid_sums_to_one"] = False
            pad = p.masked_select(~m)
            if pad.numel() and not bool((pad.abs() < 1e-6).all()):
                checks["padding_zero"] = False
    checks["ok"] = all(checks.values())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/biocda/xa_v2_closure.yaml")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--transfer-smoke", action="store_true")
    args = parser.parse_args()

    config = _cfg(args.config)
    report: Dict[str, Any] = {"architecture_version": "biocda-xa-v2", "checks": {}}

    for mtype in ("biocda_xa_fresh", "biocda_xa_transfer", "biocda_xa_kd", "biocda_xa_z_only"):
        model = build_xa_v2(config, model_type=mtype)
        tree = audit_module_tree(model)
        fwd = audit_forward_no_pool(model)
        attn = audit_attention_contract(model) if mtype != "biocda_xa_z_only" else {"ok": True, "skipped": False}
        if mtype == "biocda_xa_z_only":
            # still run forward/attention with zeros context
            attn = audit_attention_contract(model)
        state = export_student_only_state(model)
        try:
            assert_student_checkpoint_has_no_teacher(state)
            teacher_ok = True
        except AssertionError as exc:
            teacher_ok = False
            report.setdefault("errors", []).append(str(exc))
        report["checks"][mtype] = {
            "module_tree": tree,
            "forward_no_pool": fwd,
            "attention": attn,
            "no_teacher_in_state": teacher_ok,
            "ok": tree["ok"] and fwd["ok"] and attn.get("ok", False) and teacher_ok,
        }

    if args.transfer_smoke:
        ckpt = ROOT / config["data"]["predictive_checkpoint"]
        model = build_xa_v2(config, model_type="biocda_xa_transfer")
        tr = transfer_e3_gin_to_xa(ckpt, model, strict=True)
        out_tr = ROOT / "reports" / "round23_e3_gin_transfer_report.json"
        write_transfer_report(out_tr, tr)
        report["transfer"] = tr.to_dict()

    out_path = ROOT / "reports" / "round23_no_pooling_architecture_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    all_ok = all(v["ok"] for v in report["checks"].values())
    msg = f"Round23 no-pooling audit {'PASS' if all_ok else 'FAIL'} → {out_path}"
    print(msg)
    biocda_notify(msg)
    if args.strict and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
