"""Modality scale diagnostics for Z and C."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor


def modality_scale_report(
    omics_latent: Tensor,
    biological_context: Tensor,
    sample_representation: Tensor | None = None,
) -> Dict[str, Any]:
    def _stats(x: Tensor, name: str) -> Dict[str, float]:
        return {
            f"{name}_mean": float(x.mean()),
            f"{name}_std": float(x.std(unbiased=False)),
            f"{name}_norm": float(x.norm(dim=-1).mean()),
        }

    report: Dict[str, Any] = {}
    report.update(_stats(omics_latent, "Z"))
    report.update(_stats(biological_context, "C"))
    if sample_representation is not None:
        report.update(_stats(sample_representation, "sample_repr"))
    return report
