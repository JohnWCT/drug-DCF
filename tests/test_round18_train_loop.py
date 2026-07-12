import math

from tools.round18_train_loop import run_synthetic_smoke_train


def test_smoke_train_mlp_and_transformer():
    for family in ("pooled_mlp", "pooled_transformer"):
        out = run_synthetic_smoke_train(family, steps=2)
        assert math.isfinite(out["train"]["loss"])
        assert out["metrics"]["Global_AUC"] is not None


def test_smoke_train_cross_attention_modes():
    for mode in ("pure", "pooled_residual"):
        out = run_synthetic_smoke_train("cross_attention", residual_mode=mode, steps=2)
        assert math.isfinite(out["train"]["loss"])
        assert out["architecture_family"] == "cross_attention"
