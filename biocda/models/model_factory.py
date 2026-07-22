"""Model factory — M0 pooled_baseline, M1 biocda_xa_z, M2 biocda_xa_zc only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch.nn as nn

from biocda.models.biocda_model import BioCDA
from biocda.models.biological_context import SampleRepresentationZ, SampleRepresentationZC
from biocda.models.cross_attention import SampleAtomCrossAttention
from biocda.models.drug_gin import DrugGINNodeEncoder, DrugGINPooledEncoder
from biocda.models.omics_encoder import FrozenOmicsEncoder
from biocda.models.pooled_baseline import PooledBaselineModel
from biocda.models.response_head import BioCDAResponseHead
from tools.round18_response_head import Round18ResponseHead
from tools.round19_fusion_models import AdapterMLPFusion

ALLOWED_MODEL_TYPES = frozenset(
    {
        "pooled_baseline",
        "biocda_xa_z",
        "biocda_xa_zc",
        "cross_attention",
        # Round 23 XA v2 (prefer biocda.models.xa.factory for these)
        "biocda_xa_fresh",
        "biocda_xa_transfer",
        "biocda_xa_kd",
        "biocda_xa_z_only",
        "biocda_predictive",
    }
)


def _model_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["model"]


def _normalize_model_type(model_type: str) -> str:
    if model_type == "cross_attention":
        return "biocda_xa_zc"
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model type {model_type!r}; allowed: {sorted(ALLOWED_MODEL_TYPES)}"
        )
    return model_type


def export_freeze_policy(model: nn.Module, path: Path) -> Dict[str, Any]:
    groups = {
        "omics_encoder": getattr(model, "omics_encoder", None),
        "sample_encoder": getattr(model, "sample_encoder", None),
        "drug_encoder": getattr(model, "drug_encoder", None),
        "cross_attention": getattr(model, "cross_attention", None),
        "response_head": getattr(model, "response_head", None),
        "fusion": getattr(model, "fusion", None),
    }
    rows: List[Dict[str, Any]] = []
    trainable = 0
    frozen = 0
    for group_name, module in groups.items():
        if module is None:
            continue
        for name, param in module.named_parameters():
            count = int(param.numel())
            if param.requires_grad:
                trainable += count
            else:
                frozen += count
            rows.append(
                {
                    "parameter_name": f"{group_name}.{name}",
                    "shape": list(param.shape),
                    "requires_grad": bool(param.requires_grad),
                    "parameter_count": count,
                    "group_name": group_name,
                }
            )
    total = trainable + frozen
    payload = {
        "parameters": rows,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": frozen,
        "trainable_fraction": (trainable / total) if total else 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _apply_freeze_policy(model: nn.Module, config: Dict[str, Any]) -> None:
    freeze = config.get("training", {}).get("freeze_policy", {})
    mapping = {
        "omics_encoder": getattr(model, "omics_encoder", None),
        "sample_encoder": getattr(model, "sample_encoder", None),
        "drug_encoder": getattr(model, "drug_encoder", None),
        "cross_attention": getattr(model, "cross_attention", None),
        "response_head": getattr(model, "response_head", None),
        "fusion": getattr(model, "fusion", None),
    }
    for name, module in mapping.items():
        if module is None:
            continue
        should_freeze = bool(freeze.get(name, False))
        for p in module.parameters():
            p.requires_grad = not should_freeze


def _build_shared_encoders(mcfg: Dict[str, Any]):
    omics_cfg = mcfg["omics_encoder"]
    ctx_cfg = mcfg["biological_context"]
    drug_cfg = mcfg["drug_encoder"]
    omics_dim = int(omics_cfg["latent_dim"])
    context_dim = int(ctx_cfg["context_dim"])
    omics_encoder = FrozenOmicsEncoder(
        latent_dim=omics_dim,
        frozen=bool(omics_cfg.get("frozen", True)),
    )
    drug_encoder = DrugGINNodeEncoder(
        input_dim=int(drug_cfg.get("input_dim", 78)),
        node_hidden_dim=int(drug_cfg.get("node_hidden_dim", 32)),
        num_layers=int(drug_cfg.get("num_layers", 5)),
        frozen=bool(drug_cfg.get("frozen", True)),
    )
    return omics_encoder, drug_encoder, omics_dim, context_dim


def build_biocda_xa(
    config: Dict[str, Any],
    *,
    use_context_in_query: bool,
    architecture_name: str,
) -> BioCDA:
    mcfg = _model_cfg(config)
    sample_cfg = mcfg["sample_representation"]
    attn_cfg = mcfg["cross_attention"]
    head_cfg = mcfg["response_head"]

    omics_encoder, drug_encoder, omics_dim, context_dim = _build_shared_encoders(mcfg)
    sample_output_dim = sample_cfg.get("output_dim")

    if use_context_in_query:
        sample_encoder = SampleRepresentationZC(
            omics_dim=omics_dim,
            context_dim=context_dim,
            output_dim=sample_output_dim,
        )
    else:
        sample_encoder = SampleRepresentationZ(
            omics_dim=omics_dim,
            output_dim=sample_output_dim or omics_dim,
        )

    attention_dim = int(attn_cfg["attention_dim"])
    cross_attention = SampleAtomCrossAttention(
        sample_dim=sample_encoder.output_dim,
        node_dim=drug_encoder.node_dim,
        attention_dim=attention_dim,
        num_heads=int(attn_cfg["num_heads"]),
        dropout=float(attn_cfg.get("dropout", 0.1)),
        temperature=float(attn_cfg.get("temperature", 1.0)),
        use_bias=bool(attn_cfg.get("use_bias", True)),
    )
    response_head = BioCDAResponseHead(
        sample_dim=sample_encoder.output_dim,
        drug_dim=attention_dim,
        hidden_dims=tuple(head_cfg.get("hidden_dims", (256, 128))),
        dropout=float(head_cfg.get("dropout", 0.2)),
    )
    model = BioCDA(
        omics_encoder=omics_encoder,
        sample_encoder=sample_encoder,
        drug_encoder=drug_encoder,
        cross_attention=cross_attention,
        response_head=response_head,
        architecture_name=architecture_name,
    )
    _apply_freeze_policy(model, config)
    return model


def build_pooled_baseline(config: Dict[str, Any]) -> PooledBaselineModel:
    mcfg = _model_cfg(config)
    sample_cfg = mcfg["sample_representation"]
    fusion_cfg = mcfg.get("fusion", {"adapter_dim": 64})

    omics_encoder, node_encoder, omics_dim, context_dim = _build_shared_encoders(mcfg)
    sample_encoder = SampleRepresentationZC(
        omics_dim=omics_dim,
        context_dim=context_dim,
        output_dim=sample_cfg.get("output_dim"),
    )
    drug_encoder = DrugGINPooledEncoder.from_node_encoder(node_encoder)
    graph_dim = int(mcfg["drug_encoder"].get("graph_output_dim", 32))
    fusion = AdapterMLPFusion(
        omics_dim=sample_encoder.output_dim,
        drug_dim=graph_dim,
        adapter_dim=int(fusion_cfg.get("adapter_dim", 64)),
    )
    response_head = Round18ResponseHead(input_dim=fusion.output_dim)
    model = PooledBaselineModel(
        omics_encoder=omics_encoder,
        sample_encoder=sample_encoder,
        drug_encoder=drug_encoder,
        fusion=fusion,
        response_head=response_head,
    )
    _apply_freeze_policy(model, config)
    return model


def build_model(config: Dict[str, Any]) -> Union[BioCDA, PooledBaselineModel]:
    model_type = _normalize_model_type(_model_cfg(config)["type"])
    if model_type == "biocda_predictive":
        raise ValueError(
            "biocda_predictive is LOCKED_REFERENCE — use "
            "biocda.models.predictive.load_biocda_predictive(checkpoint)"
        )
    if model_type in {"biocda_xa_fresh", "biocda_xa_transfer", "biocda_xa_kd", "biocda_xa_z_only"}:
        from biocda.models.xa.factory import build_xa_v2

        return build_xa_v2(config, model_type=model_type)
    if model_type == "pooled_baseline":
        return build_pooled_baseline(config)
    if model_type == "biocda_xa_z":
        return build_biocda_xa(
            config,
            use_context_in_query=False,
            architecture_name="BioCDA-XA-Z",
        )
    if model_type == "biocda_xa_zc":
        return build_biocda_xa(
            config,
            use_context_in_query=True,
            architecture_name="BioCDA-XA-ZC",
        )
    raise ValueError(f"Unhandled model type: {model_type}")


def build_model_config_for_type(base_config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    import copy

    cfg = copy.deepcopy(base_config)
    cfg["model"]["type"] = _normalize_model_type(model_type)
    return cfg
