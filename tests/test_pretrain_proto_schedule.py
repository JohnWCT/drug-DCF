from tools.pretrain_proto_schedule import (
    compute_proto_checkpoint_guard,
    get_lambda_cmmd_eff,
    get_lambda_proto_eff,
    resolve_proto_training_params,
)


def test_lambda_proto_zero_is_noop():
    param = {"lambda_proto": 0.0, "proto_start_epoch": 5, "proto_full_epoch": 30}
    assert get_lambda_proto_eff(100, param) == 0.0


def test_schedule_rampup_values():
    param = {"lambda_proto": 0.3, "proto_start_epoch": 5, "proto_full_epoch": 30}
    assert get_lambda_proto_eff(4, param) == 0.0
    mid = get_lambda_proto_eff(17, param)
    assert 0.0 < mid < 0.3
    assert get_lambda_proto_eff(30, param) == 0.3


def test_resolve_proto_defaults():
    cfg = resolve_proto_training_params({})
    assert cfg["lambda_proto"] == 0.0
    assert cfg["proto_temperature"] == 0.2
    assert cfg["lambda_adv"] == 1.0
    assert cfg["proto_mode"] == "combined"
    assert cfg["proto_direction"] == "symmetric"
    assert cfg["proto_detach"] is True


def test_proto_checkpoint_guard_flags_early_best():
    guard = compute_proto_checkpoint_guard(
        {"lambda_proto": 0.01, "proto_start_epoch": 30, "proto_full_epoch": 50}, 10
    )
    assert guard["proto_not_effective_checkpoint"] is True


def test_proto_checkpoint_guard_ok_when_late_best():
    guard = compute_proto_checkpoint_guard(
        {"lambda_proto": 0.01, "proto_start_epoch": 5, "proto_full_epoch": 30}, 40
    )
    assert guard["proto_not_effective_checkpoint"] is False


def test_cmmd_ramp_zero_before_start():
    assert get_lambda_cmmd_eff(5, {"lambda_cmmd": 0.03, "cmmd_start_epoch": 10, "cmmd_full_epoch": 40}) == 0.0


def test_cross_domain_defaults_target_to_source():
    cfg = resolve_proto_training_params({"proto_mode": "cross_domain"})
    assert cfg["proto_direction"] == "target_to_source"
    assert cfg["proto_pair_align"] is False


def test_class_gap_schedule_ramp():
    from tools.pretrain_proto_schedule import get_lambda_class_gap_eff

    param = {"lambda_class_gap": 0.001, "class_gap_start_epoch": 5, "class_gap_full_epoch": 30}
    assert get_lambda_class_gap_eff(1, param) == 0.0
    assert get_lambda_class_gap_eff(30, param) == 0.001
