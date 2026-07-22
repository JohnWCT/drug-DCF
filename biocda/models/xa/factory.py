"""Factory for BioCDA-XA v2 candidates and predictive reference label."""
from __future__ import annotations

from typing import Any, Dict

from biocda.models.xa.cross_attention import OmicsAtomCrossAttentionStack
from biocda.models.xa.gin_atom_encoder import GINAtomEncoder
from biocda.models.xa.model import BioCDAXA
from biocda.models.xa.response_head import XAQueryResponseHead
from biocda.models.xa.sample_query import SampleQueryProjector, SampleQueryProjectorZOnly

ALLOWED = frozenset(
    {
        "biocda_xa_fresh",
        "biocda_xa_transfer",
        "biocda_xa_kd",
        "biocda_xa_z_only",
    }
)


def build_xa_v2(config: Dict[str, Any], *, model_type: str) -> BioCDAXA:
    if model_type not in ALLOWED:
        raise ValueError(f"Unknown XA model type {model_type!r}; allowed={sorted(ALLOWED)}")

    mcfg = config.get("model", config)
    xa = mcfg.get("cross_attention", {})
    drug = mcfg.get("drug_encoder", {})
    head = mcfg.get("response_head", {})

    d_model = int(xa.get("d_model", 128))
    use_context = model_type != "biocda_xa_z_only"
    if use_context:
        sample_projector = SampleQueryProjector(omics_dim=64, context_dim=32, d_model=d_model)
    else:
        sample_projector = SampleQueryProjectorZOnly(omics_dim=64, d_model=d_model)

    drug_encoder = GINAtomEncoder(
        input_dim=int(drug.get("input_dim", 78)),
        node_hidden_dim=int(drug.get("node_hidden_dim", 32)),
        num_layers=int(drug.get("num_layers", 5)),
        jk_mode=str(drug.get("jk_mode", "last")),
        dropout=float(drug.get("dropout", 0.1)),
        use_batch_norm=bool(drug.get("use_batch_norm", True)),
    )
    cross_attention = OmicsAtomCrossAttentionStack(
        d_model=d_model,
        num_heads=int(xa.get("num_heads", 4)),
        num_layers=int(xa.get("num_layers", 2)),
        ffn_dim=int(xa.get("ffn_dim", 256)),
        attention_dropout=float(xa.get("attention_dropout", 0.1)),
        block_dropout=float(xa.get("block_dropout", 0.2)),
        node_dim=drug_encoder.node_dim,
    )
    response_head = XAQueryResponseHead(
        d_model=d_model,
        hidden_dim=int(head.get("hidden_dim", 128)),
        dropout=float(head.get("dropout", 0.1)),
    )
    return BioCDAXA(
        sample_projector=sample_projector,
        drug_encoder=drug_encoder,
        cross_attention=cross_attention,
        response_head=response_head,
        use_context=use_context,
    )


def build_model(config: Dict[str, Any]) -> BioCDAXA:
    model_type = config["model"]["type"]
    if model_type == "biocda_predictive":
        raise ValueError(
            "biocda_predictive is LOCKED_REFERENCE — load Round20 checkpoints; "
            "do not instantiate via XA factory"
        )
    return build_xa_v2(config, model_type=model_type)
