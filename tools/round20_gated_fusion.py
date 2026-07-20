"""Round 20 Stage 20B: gated pooled fusion predictor.

Fuses a frozen O2 omics vector with the D0 pooled drug embedding via a learned
per-dimension gate, then predicts response. The module is compatible with the
Round 19 ``forward_round19_batch`` P0 code path: it is called as
``fusion(omics, drug_embedding, return_interpretability=..., return_attention=...)``
and returns logits so that an ``nn.Identity`` head can follow.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedPooledFusionPredictor(nn.Module):
    def __init__(
        self,
        omics_dim: int,
        drug_dim: int = 32,
        hidden_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if omics_dim <= 0:
            raise ValueError("omics_dim must be positive")
        if drug_dim <= 0:
            raise ValueError("drug_dim must be positive")

        self.omics_dim = int(omics_dim)
        self.drug_dim = int(drug_dim)
        self.hidden_dim = int(hidden_dim)
        # Exposed so an external head could be attached; here logits are produced internally.
        self.output_dim = 1

        self.omics_projection = nn.Sequential(
            nn.Linear(omics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.drug_projection = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        omics: Tensor,
        drug_embedding: Tensor,
        *,
        return_gate: bool = False,
        return_interpretability: bool = False,
        return_attention: bool = False,
    ):
        if return_interpretability or return_attention:
            raise ValueError("GatedPooledFusion has no atom-level interpretability")
        if omics.ndim != 2:
            raise ValueError(f"Expected omics shape [B, D], received {tuple(omics.shape)}")
        if drug_embedding.ndim != 2:
            raise ValueError(
                f"Expected drug embedding shape [B, D], received {tuple(drug_embedding.shape)}"
            )
        if omics.shape[0] != drug_embedding.shape[0]:
            raise ValueError("Omics and drug batch sizes differ")
        if omics.shape[-1] != self.omics_dim:
            raise ValueError(f"Expected omics dim {self.omics_dim}, received {omics.shape[-1]}")
        if drug_embedding.shape[-1] != self.drug_dim:
            raise ValueError(f"Expected drug dim {self.drug_dim}, received {drug_embedding.shape[-1]}")

        omics_h = self.omics_projection(omics)
        drug_h = self.drug_projection(drug_embedding)
        gate = self.gate_network(torch.cat([omics_h, drug_h], dim=-1))
        fused = gate * drug_h + (1.0 - gate) * omics_h
        logits = self.prediction_head(fused).squeeze(-1)
        if return_gate:
            return logits, gate
        return logits
