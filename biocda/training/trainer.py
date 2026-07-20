"""Minimal BioCDA trainer with GPU-efficient defaults (architecture phase)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from biocda.utils.gpu import build_efficient_dataloader_kwargs, configure_gpu_efficiency


@dataclass
class TrainStepStats:
    loss: float
    device: str


class BioCDATrainer:
    """Small trainer shell; full drug-held-out loop comes in a later round."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: Optional[str] = None,
        use_amp: bool = True,
        micro_batch_size: int = 256,
    ) -> None:
        configure_gpu_efficiency(target_utilization=0.9)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.use_amp = bool(use_amp and self.device.startswith("cuda"))
        self.scaler = GradScaler(enabled=self.use_amp)
        self.micro_batch_size = int(micro_batch_size)
        self.dataloader_defaults = build_efficient_dataloader_kwargs(
            batch_size=self.micro_batch_size,
        )

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
    ) -> TrainStepStats:
        self.model.train()
        total = 0.0
        n = 0
        for batch in dataloader:
            omics = batch["omics"].to(self.device, non_blocking=True)
            context = batch["context"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            drug_graph = batch["drug_graph"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                out = self.model(omics, context, drug_graph, output_mode="prediction")
                loss = loss_fn(out.logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach())
            n += 1
        return TrainStepStats(loss=total / max(n, 1), device=self.device)
