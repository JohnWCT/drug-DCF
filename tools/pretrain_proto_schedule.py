"""Prototype InfoNCE and class-wise MMD schedule helpers (testable without CUDA)."""

from __future__ import annotations

from typing import Optional


def smooth_rampup(epoch: int, start_epoch: int, end_epoch: int, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return float(max_value) if epoch >= start_epoch else 0.0
    if epoch >= end_epoch:
        return float(max_value)
    phase = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(max_value) * (phase ** 3)


def get_lambda_proto_eff(gan_epoch: int, param: dict) -> float:
    lambda_proto = float(param.get("lambda_proto", 0.0))
    if lambda_proto <= 0:
        return 0.0
    proto_start_epoch = int(param.get("proto_start_epoch", 1))
    proto_full_epoch = int(param.get("proto_full_epoch", proto_start_epoch))
    return smooth_rampup(gan_epoch, proto_start_epoch, proto_full_epoch, lambda_proto)


def get_lambda_class_gap_eff(gan_epoch: int, param: dict) -> float:
    lambda_class_gap = float(param.get("lambda_class_gap", 0.0))
    if lambda_class_gap <= 0:
        return 0.0
    start_epoch = int(param.get("class_gap_start_epoch", 5))
    full_epoch = int(param.get("class_gap_full_epoch", start_epoch))
    return smooth_rampup(gan_epoch, start_epoch, full_epoch, lambda_class_gap)


def get_lambda_cmmd_eff(gan_epoch: int, param: dict) -> float:
    lambda_cmmd = float(param.get("lambda_cmmd", 0.0))
    if lambda_cmmd <= 0:
        return 0.0
    cmmd_start_epoch = int(param.get("cmmd_start_epoch", 1))
    cmmd_full_epoch = int(param.get("cmmd_full_epoch", cmmd_start_epoch))
    return smooth_rampup(gan_epoch, cmmd_start_epoch, cmmd_full_epoch, lambda_cmmd)


def resolve_proto_training_params(param: dict) -> dict:
    proto_mode = str(param.get("proto_mode", "combined"))
    direction_default = "target_to_source" if proto_mode == "cross_domain" else "symmetric"
    return {
        "lambda_proto": float(param.get("lambda_proto", 0.0)),
        "proto_temperature": float(param.get("proto_temperature", 0.2)),
        "proto_start_epoch": int(param.get("proto_start_epoch", 1)),
        "proto_full_epoch": int(param.get("proto_full_epoch", 1)),
        "proto_min_samples_per_class": int(
            param.get("proto_min_samples_per_class", param.get("min_proto_samples_per_class", 1))
        ),
        "proto_min_samples_per_domain": int(param.get("proto_min_samples_per_domain", 1)),
        "proto_mode": proto_mode,
        "proto_direction": str(param.get("proto_direction", direction_default)),
        "proto_detach": bool(param.get("proto_detach", True)),
        "proto_pair_align": bool(param.get("proto_pair_align", False)),
        "lambda_proto_pair": float(param.get("lambda_proto_pair", 0.0)),
        "proto_pair_temperature": float(param.get("proto_pair_temperature", 1.0)),
        "lambda_adv": float(param.get("lambda_adv", 1.0)),
    }


def post_proto_checkpoint_min_epoch(param: dict) -> int:
    """Minimum GAN epoch at which post-proto checkpoint is eligible."""
    proto_start = int(param.get("proto_start_epoch", 1))
    proto_full = int(param.get("proto_full_epoch", proto_start))
    return max(proto_start + 5, proto_full)


def resolve_class_gap_training_params(param: dict) -> dict:
    return {
        "lambda_class_gap": float(param.get("lambda_class_gap", 0.0)),
        "class_gap_metric": str(param.get("class_gap_metric", "cosine")),
        "class_gap_start_epoch": int(param.get("class_gap_start_epoch", 5)),
        "class_gap_full_epoch": int(param.get("class_gap_full_epoch", 30)),
        "class_gap_min_samples_per_domain": int(param.get("class_gap_min_samples_per_domain", 2)),
        "class_gap_detach_source": bool(param.get("class_gap_detach_source", True)),
        "class_gap_detach_target": bool(param.get("class_gap_detach_target", False)),
        "class_gap_l2_squared": bool(param.get("class_gap_l2_squared", True)),
    }


def resolve_cmmd_training_params(param: dict) -> dict:
    return {
        "lambda_cmmd": float(param.get("lambda_cmmd", 0.0)),
        "cmmd_start_epoch": int(param.get("cmmd_start_epoch", 10)),
        "cmmd_full_epoch": int(param.get("cmmd_full_epoch", 40)),
        "cmmd_min_samples_per_domain": int(param.get("cmmd_min_samples_per_domain", 2)),
        "cmmd_gamma": param.get("cmmd_gamma", "median"),
    }


def compute_proto_checkpoint_guard(
    param: dict,
    best_gan_epoch_overall: int,
    best_gan_epoch_post_proto: int = 0,
    best_gan_loss_overall: Optional[float] = None,
    best_gan_loss_post_proto: Optional[float] = None,
) -> dict:
    """Flag checkpoints where InfoNCE ramp had not started before best overall GAN epoch."""
    lambda_proto = float(param.get("lambda_proto", 0.0))
    proto_start_epoch = int(param.get("proto_start_epoch", 1))
    proto_full_epoch = int(param.get("proto_full_epoch", proto_start_epoch))
    min_post = post_proto_checkpoint_min_epoch(param)

    post_available = int(best_gan_epoch_post_proto) >= min_post
    proto_not_effective_overall = bool(lambda_proto > 0 and int(best_gan_epoch_overall) < proto_start_epoch)

    if lambda_proto > 0:
        selection_epoch = int(best_gan_epoch_post_proto) if post_available else 0
        selection_type = "post_proto" if post_available else "none"
        proto_invalid = not post_available
    else:
        selection_epoch = int(best_gan_epoch_overall)
        selection_type = "overall"
        proto_invalid = False

    return {
        "proto_not_effective_checkpoint": proto_not_effective_overall,
        "proto_effective_checkpoint_available": post_available if lambda_proto > 0 else True,
        "proto_invalid": proto_invalid,
        "proto_effective_epoch": selection_epoch if selection_epoch > 0 else None,
        "proto_start_epoch": proto_start_epoch,
        "proto_full_epoch": proto_full_epoch,
        "post_proto_checkpoint_min_epoch": min_post,
        "best_gan_epoch": selection_epoch if lambda_proto > 0 and post_available else int(best_gan_epoch_overall),
        "best_gan_epoch_overall": int(best_gan_epoch_overall),
        "best_gan_epoch_post_proto": int(best_gan_epoch_post_proto),
        "best_gan_loss_overall": best_gan_loss_overall,
        "best_gan_loss_post_proto": best_gan_loss_post_proto,
        "selection_checkpoint_type": selection_type,
    }
