"""D0-Pooled baseline for architecture comparison."""
from __future__ import annotations

from typing import Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from biocda.models.outputs import BioCDAOutput
from biocda.models.response_head import BioCDAResponseHead
from tools.round18_response_head import Round18ResponseHead
from tools.round19_fusion_models import AdapterMLPFusion


class PooledBaselineModel(nn.Module):
    """D0-Pooled: global max-pooled graph embedding + adapter fusion (no atom attention)."""

    ARCHITECTURE_VERSION = "d0-pooled-v1"
    VALID_OUTPUT_MODES = frozenset({"prediction", "full"})

    def __init__(
        self,
        omics_encoder: nn.Module,
        sample_encoder: nn.Module,
        drug_encoder: nn.Module,
        fusion: AdapterMLPFusion,
        response_head: Round18ResponseHead,
    ) -> None:
        super().__init__()
        self.omics_encoder = omics_encoder
        self.sample_encoder = sample_encoder
        self.drug_encoder = drug_encoder
        self.fusion = fusion
        self.response_head = response_head

    def forward(
        self,
        omics: torch.Tensor,
        biological_context: torch.Tensor,
        drug_graph: Union[Data, Batch],
        *,
        output_mode: str = "prediction",
    ) -> BioCDAOutput:
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(
                f"Pooled baseline supports output_mode in {self.VALID_OUTPUT_MODES}, got {output_mode!r}"
            )
        omics_latent = self.omics_encoder(omics)
        sample_repr = self.sample_encoder(omics_latent, biological_context)
        drug_pooled = self.drug_encoder(drug_graph)
        fusion_repr = self.fusion(sample_repr, drug_pooled)
        logits = self.response_head(fusion_repr)
        probabilities = torch.sigmoid(logits)
        if output_mode == "prediction":
            return BioCDAOutput(logits=logits, probabilities=probabilities)
        return BioCDAOutput(
            logits=logits,
            probabilities=probabilities,
            sample_representation=sample_repr,
            omics_latent=omics_latent,
            biological_context=biological_context,
            drug_representation=drug_pooled,
        )
