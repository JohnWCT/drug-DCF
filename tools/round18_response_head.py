"""Fixed Round 18 response head shared across all fusion architectures."""
from __future__ import annotations

from torch import Tensor, nn


class Round18ResponseHead(nn.Module):
    """fusion_repr -> Linear -> ReLU -> Dropout -> Linear(logit)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)
