"""Prototype InfoNCE schedule helpers (testable without CUDA)."""


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


def resolve_proto_training_params(param: dict) -> dict:
    return {
        "lambda_proto": float(param.get("lambda_proto", 0.0)),
        "proto_temperature": float(param.get("proto_temperature", 0.2)),
        "proto_start_epoch": int(param.get("proto_start_epoch", 1)),
        "proto_full_epoch": int(param.get("proto_full_epoch", 1)),
        "proto_min_samples_per_class": int(
            param.get("proto_min_samples_per_class", param.get("min_proto_samples_per_class", 1))
        ),
        "lambda_adv": float(param.get("lambda_adv", 1.0)),
    }
