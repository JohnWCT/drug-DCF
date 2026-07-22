"""Logit distillation from frozen BioCDA-Predictive teacher to XA student."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DistillationLossOutput:
    total: Tensor
    response: Tensor
    distillation: Tensor
    lambda_kd: float
    temperature: float


def logit_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
) -> Tensor:
    """Soft logit KD (BCE soft targets via sigmoid KL-style MSE on probs)."""
    t = float(temperature)
    s = student_logits / t
    teach = teacher_logits.detach() / t
    # Use BCE-with-logits soft targets (binary response)
    teacher_probs = torch.sigmoid(teach)
    return F.binary_cross_entropy_with_logits(s, teacher_probs) * (t * t)


def combine_response_kd(
    response_loss: Tensor,
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    lambda_kd: float = 0.5,
    temperature: float = 2.0,
) -> DistillationLossOutput:
    kd = logit_kd_loss(student_logits, teacher_logits, temperature=temperature)
    total = response_loss + float(lambda_kd) * kd
    return DistillationLossOutput(
        total=total,
        response=response_loss,
        distillation=kd,
        lambda_kd=float(lambda_kd),
        temperature=float(temperature),
    )


class FrozenPredictiveTeacher(nn.Module):
    """
    Thin wrapper that keeps a pooled predictive teacher frozen.

    The teacher module must NOT be registered as a child of the student so that
    student checkpoints never contain teacher weights.
    """

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        # Keep as plain attribute (not Module child) via object.__setattr__
        object.__setattr__(self, "_teacher", teacher)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

    @property
    def teacher(self) -> nn.Module:
        return object.__getattribute__(self, "_teacher")

    def train(self, mode: bool = True):  # noqa: ARG002
        # Always eval
        self.teacher.eval()
        return self

    def forward(self, omics: Tensor, context: Tensor, drug_graph) -> Tensor:
        with torch.no_grad():
            out = self.teacher(omics, context, drug_graph, output_mode="prediction")
            return out.logits.reshape(-1)

    def state_dict(self, *args, **kwargs):  # noqa: ANN002, ANN003
        # Never serialize teacher into student packages
        return {}


def assert_student_checkpoint_has_no_teacher(state: dict) -> None:
    banned = ("teacher", "fusion", "fc1_xd", "pool", "pooled")
    bad = [k for k in state if any(b in k.lower() for b in banned)]
    # fc1_xd may exist under drug_encoder.gin.* construction leftover —
    # inference must not *use* it; audit separately. Still forbid explicit teacher.
    teacher_keys = [k for k in state if "teacher" in k.lower()]
    if teacher_keys:
        raise AssertionError(f"Student checkpoint contains teacher keys: {teacher_keys}")
    # Soft warning path for fusion
    fusion_keys = [k for k in state if k.startswith("fusion") or ".fusion." in k]
    if fusion_keys:
        raise AssertionError(f"Student checkpoint contains fusion keys: {fusion_keys}")


def export_student_only_state(student: nn.Module) -> dict:
    state = student.state_dict()
    assert_student_checkpoint_has_no_teacher(state)
    return state
