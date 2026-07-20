"""GPU efficiency helpers for BioCDA training and inference."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def configure_gpu_efficiency(*, target_utilization: float = 0.9) -> Dict[str, Any]:
    """Apply common PyTorch GPU throughput settings.

    Round 20 dispatch packs many light jobs (~90% system throughput). BioCDA
    training should prefer large micro-batches + pinned memory + cudnn benchmark.
    """
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "target_utilization": target_utilization,
    }
    if not torch.cuda.is_available():
        return info

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    device_count = torch.cuda.device_count()
    info["device_count"] = device_count
    info["cudnn_benchmark"] = True
    return info


def recommend_dataloader_workers(*, batch_size: int, cpu_count: Optional[int] = None) -> int:
    """Heuristic worker count to keep GPU fed without oversubscribing CPU."""
    cpus = cpu_count or os.cpu_count() or 4
    # More workers for small batches (graph collation bound); cap to leave headroom.
    if batch_size <= 64:
        return min(max(cpus - 2, 4), 12)
    return min(max(cpus // 2, 2), 8)


def build_efficient_dataloader_kwargs(
    *,
    batch_size: int,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    use_pin = pin_memory and torch.cuda.is_available()
    return {
        "batch_size": batch_size,
        "num_workers": recommend_dataloader_workers(batch_size=batch_size),
        "pin_memory": use_pin,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
