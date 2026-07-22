"""Query / drug utilization diagnostics for XA v2."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def query_residual_dominance(
    initial_query: Tensor,
    final_query: Tensor,
) -> Dict[str, float]:
    """cosine(Q0, Qfinal) — near 1 may indicate drug info not injected."""
    q0 = initial_query[:, 0, :]
    qf = final_query[:, 0, :]
    cos = F.cosine_similarity(q0, qf, dim=-1)
    return {
        "mean_q0_qfinal_cosine": float(cos.mean()),
        "median_q0_qfinal_cosine": float(cos.median()),
        "min_q0_qfinal_cosine": float(cos.min()),
        "q0_norm_mean": float(q0.norm(dim=-1).mean()),
        "qfinal_norm_mean": float(qf.norm(dim=-1).mean()),
    }


@torch.no_grad()
def drug_replacement_sensitivity(
    model: nn.Module,
    omics: Tensor,
    context: Tensor,
    drug_a,
    drug_b,
) -> Dict[str, float]:
    """Swap drug graphs; prediction should change if drug path is used."""
    model.eval()
    out_a = model(omics, context, drug_a, output_mode="full")
    out_b = model(omics, context, drug_b, output_mode="full")
    logit_delta = (out_a.logits - out_b.logits).abs()
    atom_delta = (out_a.dense_atom_tokens.mean(dim=1) - out_b.dense_atom_tokens.mean(dim=1)).norm(dim=-1)
    return {
        "mean_abs_logit_delta": float(logit_delta.mean()),
        "max_abs_logit_delta": float(logit_delta.max()),
        "mean_atom_token_delta": float(atom_delta.mean()),
        "pass_drug_changes_prediction": bool(logit_delta.mean() > 1e-4),
    }


@torch.no_grad()
def attention_intervention_suite(
    model: nn.Module,
    omics: Tensor,
    context: Tensor,
    drug_graph,
) -> Dict[str, Any]:
    """uniform / zero-ish override checks (faithfulness interfaces)."""
    model.eval()
    base = model(omics, context, drug_graph, output_mode="attention")
    mask = base.atom_mask  # [B,N]
    b, n = mask.shape
    h = model.cross_attention.num_heads

    # Uniform over valid atoms
    uniform = mask.float().unsqueeze(1).unsqueeze(2).expand(b, h, 1, n)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    out_u = model(
        omics,
        context,
        drug_graph,
        output_mode="prediction",
        attention_override=uniform,
    )
    delta_u = (base.logits - out_u.logits).abs()
    return {
        "mean_abs_logit_delta_uniform": float(delta_u.mean()),
        "uniform_path_runs": True,
        "pass_attention_affects_logits": bool(delta_u.mean() > 1e-5),
    }


def utilization_report(
    *,
    residual: Dict[str, float],
    drug: Dict[str, float],
    attention: Dict[str, Any],
) -> Dict[str, Any]:
    pass_query = residual["mean_q0_qfinal_cosine"] < 0.999
    pass_drug = bool(drug.get("pass_drug_changes_prediction", False))
    pass_attn = bool(attention.get("pass_attention_affects_logits", False))
    return {
        "query_residual": residual,
        "drug_sensitivity": drug,
        "attention_intervention": attention,
        "checks": {
            "query_sensitivity": pass_query,
            "drug_sensitivity": pass_drug,
            "attention_path": pass_attn,
        },
        "pass": pass_query and pass_drug and pass_attn,
    }
