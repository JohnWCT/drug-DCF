"""Phased freeze schedule for BioCDA-XA v2 training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from torch import nn


@dataclass
class FreezePhase:
    name: str
    epochs: int
    freeze_gin_layers: List[int]  # 0-indexed layers to freeze
    last_gin_lr: Optional[float] = None
    other_lr: float = 3e-4


DEFAULT_PHASES = [
    FreezePhase(name="attention_warmup", epochs=15, freeze_gin_layers=[0, 1, 2, 3, 4], other_lr=3e-4),
    FreezePhase(
        name="last_gin_adaptation",
        epochs=40,
        freeze_gin_layers=[0, 1, 2, 3],
        last_gin_lr=1e-5,
        other_lr=3e-4,
    ),
    FreezePhase(
        name="joint_stabilization",
        epochs=145,
        freeze_gin_layers=[0, 1, 2, 3],
        last_gin_lr=1e-5,
        other_lr=3e-4,
    ),
]


def _set_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_gin_except_layers(gin: nn.Module, trainable_layers: Iterable[int]) -> None:
    trainable = set(int(i) for i in trainable_layers)
    for i, conv in enumerate(gin.convs):
        _set_requires_grad(conv, i in trainable)
    if getattr(gin, "use_batch_norm", False) and gin.bns is not None:
        for i, bn in enumerate(gin.bns):
            if bn is None:
                continue
            _set_requires_grad(bn, i in trainable)


def set_frozen_bn_eval(gin: nn.Module, frozen_layers: Iterable[int]) -> None:
    """Frozen GIN BatchNorm must stay in eval so running stats do not drift."""
    frozen = set(int(i) for i in frozen_layers)
    if not getattr(gin, "use_batch_norm", False):
        return
    for i, bn in enumerate(gin.bns):
        if bn is None:
            continue
        if i in frozen:
            bn.eval()


def apply_phase(model: nn.Module, phase: FreezePhase) -> Dict[str, Any]:
    gin = model.drug_encoder.gin
    n_layers = int(gin.num_layers)
    frozen = [i for i in phase.freeze_gin_layers if 0 <= i < n_layers]
    trainable = [i for i in range(n_layers) if i not in frozen]

    # Always train projector / CA / head
    _set_requires_grad(model.sample_projector, True)
    _set_requires_grad(model.cross_attention, True)
    _set_requires_grad(model.response_head, True)

    freeze_gin_except_layers(gin, trainable)
    set_frozen_bn_eval(gin, frozen)

    return {
        "phase": phase.name,
        "frozen_gin_layers": frozen,
        "trainable_gin_layers": trainable,
        "last_gin_lr": phase.last_gin_lr,
        "other_lr": phase.other_lr,
    }


def build_param_groups(model: nn.Module, phase: FreezePhase) -> List[Dict[str, Any]]:
    gin = model.drug_encoder.gin
    n_layers = int(gin.num_layers)
    frozen = set(i for i in phase.freeze_gin_layers if 0 <= i < n_layers)
    last_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_last_gin = False
        for i in range(n_layers):
            if i in frozen:
                continue
            if f"drug_encoder.gin.convs.{i}." in name or f"drug_encoder.gin.bns.{i}." in name:
                is_last_gin = True
                break
        if is_last_gin and phase.last_gin_lr is not None:
            last_params.append(p)
        else:
            other_params.append(p)

    groups = [{"params": other_params, "lr": float(phase.other_lr)}]
    if last_params and phase.last_gin_lr is not None:
        groups.append({"params": last_params, "lr": float(phase.last_gin_lr)})
    return groups


def phase_for_epoch(epoch: int, phases: Optional[List[FreezePhase]] = None) -> FreezePhase:
    schedule = phases or DEFAULT_PHASES
    cursor = 0
    for phase in schedule:
        cursor += int(phase.epochs)
        if epoch < cursor:
            return phase
    return schedule[-1]
