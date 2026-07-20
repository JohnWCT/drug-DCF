"""Model factory for BioCDA-XA and D0-Pooled baseline."""
from __future__ import annotations

from typing import Any, Dict, Union

import torch.nn as nn

from biocda.models.biocda_model import BioCDA
from biocda.models.biological_context import SampleRepresentation
from biocda.models.cross_attention import SampleAtomCrossAttention
from biocda.models.drug_gin import DrugGINNodeEncoder, DrugGINPooledEncoder
from biocda.models.omics_encoder import FrozenOmicsEncoder
from biocda.models.pooled_baseline import PooledBaselineModel
from biocda.models.response_head import BioCDAResponseHead
from tools.round18_response_head import Round18ResponseHead
from tools.round19_fusion_models import AdapterMLPFusion


def _model_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["model"]


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


def build_cross_attention_model(config: Dict[str, Any]) -> BioCDA:
    mcfg = _model_cfg(config)
    omics_cfg = mcfg["omics_encoder"]
    ctx_cfg = mcfg["biological_context"]
    sample_cfg = mcfg["sample_representation"]
    drug_cfg = mcfg["drug_encoder"]
    attn_cfg = mcfg["cross_attention"]
    head_cfg = mcfg["response_head"]

    omics_dim = int(omics_cfg["latent_dim"])
    context_dim = int(ctx_cfg["context_dim"])
    sample_output_dim = sample_cfg.get("output_dim")
    node_dim = int(drug_cfg.get("node_dim", drug_cfg.get("node_hidden_dim", 32)))
    attention_dim = int(attn_cfg["attention_dim"])

    omics_encoder = FrozenOmicsEncoder(
        latent_dim=omics_dim,
        frozen=bool(omics_cfg.get("frozen", True)),
    )
    sample_encoder = SampleRepresentation(
        omics_dim=omics_dim,
        context_dim=context_dim,
        output_dim=sample_output_dim,
    )
    drug_encoder = DrugGINNodeEncoder(
        input_dim=int(drug_cfg.get("input_dim", 78)),
        node_hidden_dim=int(drug_cfg.get("node_hidden_dim", 32)),
        num_layers=int(drug_cfg.get("num_layers", 5)),
        frozen=bool(drug_cfg.get("frozen", True)),
    )
    cross_attention = SampleAtomCrossAttention(
        sample_dim=sample_encoder.output_dim,
        node_dim=drug_encoder.node_dim,
        attention_dim=attention_dim,
        num_heads=int(attn_cfg["num_heads"]),
        dropout=float(attn_cfg.get("dropout", 0.1)),
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
    )
    _apply_freeze_policy(model, config)
    return model


def build_pooled_baseline(config: Dict[str, Any]) -> PooledBaselineModel:
    mcfg = _model_cfg(config)
    omics_cfg = mcfg["omics_encoder"]
    ctx_cfg = mcfg["biological_context"]
    sample_cfg = mcfg["sample_representation"]
    drug_cfg = mcfg["drug_encoder"]
    fusion_cfg = mcfg.get("fusion", {"adapter_dim": 64})

    omics_dim = int(omics_cfg["latent_dim"])
    context_dim = int(ctx_cfg["context_dim"])
    sample_output_dim = sample_cfg.get("output_dim")

    omics_encoder = FrozenOmicsEncoder(
        latent_dim=omics_dim,
        frozen=bool(omics_cfg.get("frozen", True)),
    )
    sample_encoder = SampleRepresentation(
        omics_dim=omics_dim,
        context_dim=context_dim,
        output_dim=sample_output_dim,
    )
    node_encoder = DrugGINNodeEncoder(
        node_hidden_dim=int(drug_cfg.get("node_hidden_dim", 32)),
        frozen=bool(drug_cfg.get("frozen", True)),
    )
    drug_encoder = DrugGINPooledEncoder.from_node_encoder(node_encoder)
    graph_dim = int(drug_cfg.get("graph_output_dim", 32))
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
    model_type = _model_cfg(config)["type"]
    if model_type == "pooled_baseline":
        return build_pooled_baseline(config)
    if model_type == "cross_attention":
        return build_cross_attention_model(config)
    raise ValueError(f"Unknown model type: {model_type}")
