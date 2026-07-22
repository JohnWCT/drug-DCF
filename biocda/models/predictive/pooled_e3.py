"""BioCDA-Predictive (pooled E3) teacher/reference loader — Round20 LOCKED."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from drugmodels.ginconv import GINConvNet
from tools.round18_response_head import Round18ResponseHead
from tools.round19_fusion_models import AdapterMLPFusion


class BioCDAPredictive(nn.Module):
    """
    Round20 C32 + pooled E3 reference.

    LOCKED_REFERENCE — do not modify architecture; used as teacher / P0 baseline.
    """

    ARCHITECTURE_NAME = "BioCDA-Predictive"
    ARCHITECTURE_VERSION = "biocda-predictive-e3"
    STATUS = "LOCKED_REFERENCE"

    def __init__(
        self,
        *,
        omics_dim: int = 64,
        context_dim: int = 32,
        node_hidden_dim: int = 32,
        graph_output_dim: int = 32,
        adapter_dim: int = 64,
        num_layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.omics_dim = omics_dim
        self.context_dim = context_dim
        sample_dim = omics_dim + context_dim
        self.encoder = GINConvNet(
            input_dim=78,
            node_hidden_dim=node_hidden_dim,
            graph_output_dim=graph_output_dim,
            dropout=dropout,
            num_layers=num_layers,
            jk_mode="last",
            pool_type="max",
            use_batch_norm=True,
        )
        self.fusion = AdapterMLPFusion(
            omics_dim=sample_dim,
            drug_dim=graph_output_dim,
            adapter_dim=adapter_dim,
        )
        self.head = Round18ResponseHead(input_dim=self.fusion.output_dim)

    def forward(
        self,
        omics: torch.Tensor,
        biological_context: torch.Tensor,
        drug_graph: Union[Data, Batch],
        *,
        output_mode: str = "prediction",
    ):
        del output_mode
        assert omics.shape[-1] == self.omics_dim
        assert biological_context.shape[-1] == self.context_dim
        sample = torch.cat([omics, biological_context], dim=-1)
        graph_emb = self.encoder(drug_graph)
        fused = self.fusion(sample, graph_emb)
        logits = self.head(fused).reshape(-1)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.probabilities = torch.sigmoid(logits)
        return out


def load_biocda_predictive(
    checkpoint_path: Path | str,
    *,
    map_location: str = "cpu",
) -> BioCDAPredictive:
    """Load Round20 A_C32_E3 checkpoint. Omics API is Z64 + C32 (concat → 96 for fusion)."""
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=map_location)
    model = BioCDAPredictive(
        omics_dim=64,
        context_dim=32,
        node_hidden_dim=int(ckpt.get("node_hidden_dim", 32)),
        graph_output_dim=int(ckpt.get("graph_output_dim", 32)),
    )
    msd = ckpt["model_state_dict"]
    model.encoder.load_state_dict(msd["encoder"], strict=True)
    model.fusion.load_state_dict(msd["fusion"], strict=True)
    model.head.load_state_dict(msd["head"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
