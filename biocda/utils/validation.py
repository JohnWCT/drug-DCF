"""Architecture validation helpers."""
from __future__ import annotations

import torch


def assert_no_nan_inf(tensor: torch.Tensor, *, name: str) -> None:
    if torch.isnan(tensor).any():
        raise AssertionError(f"{name} contains NaN")
    if torch.isinf(tensor).any():
        raise AssertionError(f"{name} contains Inf")
