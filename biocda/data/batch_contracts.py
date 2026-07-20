"""Batch tensor contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BioCDABatch:
    omics: torch.Tensor
    biological_context: torch.Tensor
    labels: Optional[torch.Tensor] = None
