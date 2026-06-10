from tools.pretrain_proto_schedule import get_lambda_proto_eff, resolve_proto_training_params


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
