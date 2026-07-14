"""Round 19 drug encoders (MACCS-only; GIN/GINE live under drugmodels/)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from drugmodels.ginconv import GINConvNet
from drugmodels.gineconv import GINEConvNet


class MACCSDrugEncoder(nn.Module):
    """Fingerprint-only drug encoder (Round 19 D4). No graph parameters."""

    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def assert_no_hybrid(encoder_type: str, has_maccs: bool, has_graph: bool) -> None:
    enc = str(encoder_type).lower()
    if enc in {"maccs", "maccs_only"} and has_graph:
        raise AssertionError("Hybrid forbidden: MACCS encoder cannot use graph inputs")
    if enc in {"gin", "gine"} and has_maccs:
        raise AssertionError("Hybrid forbidden: GIN/GINE cannot use MACCS inputs")
    if has_maccs and has_graph:
        raise AssertionError("Hybrid forbidden: MACCS + graph in same job")


def build_drug_encoder(
    encoder_type: str,
    *,
    node_hidden_dim: int = 32,
    graph_output_dim: int = 32,
    edge_dim: int = 10,
    maccs_input_dim: int = 166,
    maccs_output_dim: int = 64,
    dropout: float = 0.1,
    num_layers: int = 5,
    jk_mode: str = "last",
    pool_type: str = "max",
) -> nn.Module:
    enc = str(encoder_type).lower()
    if enc == "gin":
        return GINConvNet(
            input_dim=78,
            node_hidden_dim=node_hidden_dim,
            graph_output_dim=graph_output_dim,
            dropout=dropout,
            num_layers=num_layers,
            jk_mode=jk_mode,
            pool_type=pool_type,
        )
    if enc == "gine":
        return GINEConvNet(
            input_dim=78,
            edge_dim=edge_dim,
            node_hidden_dim=node_hidden_dim,
            graph_output_dim=graph_output_dim,
            dropout=dropout,
            num_layers=num_layers,
            jk_mode=jk_mode,
            pool_type=pool_type,
        )
    if enc in {"maccs", "maccs_only"}:
        return MACCSDrugEncoder(
            input_dim=maccs_input_dim,
            output_dim=maccs_output_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown encoder_type={encoder_type!r}")
