"""Response head — final updated query only."""
from __future__ import annotations

from torch import Tensor, nn


class XAQueryResponseHead(nn.Module):
    """Maps Qfinal[:,0,:] → logit. No concat with pooled/raw features."""

    def __init__(self, d_model: int = 128, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.input_dim = int(d_model)

    def forward(self, final_query: Tensor) -> Tensor:
        # Accept [B,1,d] or [B,d]
        if final_query.ndim == 3:
            if final_query.shape[1] != 1:
                raise ValueError(f"Expected query length 1, got {final_query.shape}")
            head_input = final_query[:, 0, :]
        else:
            head_input = final_query
        if head_input.shape[-1] != self.input_dim:
            raise ValueError(f"Expected head input dim {self.input_dim}, got {head_input.shape[-1]}")
        return self.net(head_input).reshape(-1)
