"""Training-flag smoke tests for Round 12 prototype alignment wiring."""

import inspect

import pretrain_VAEwC as pv
from tools.source_anchor_prototypes import resolve_source_anchor_proto_training_params


def test_resolve_disabled_by_default():
    cfg = resolve_source_anchor_proto_training_params({})
    assert cfg["source_anchor_proto_enabled"] is False
    assert cfg["lambda_proto_align"] == 0.0


def test_resolve_enabled_params():
    cfg = resolve_source_anchor_proto_training_params(
        {
            "source_anchor_proto_enabled": True,
            "lambda_proto_align": 0.001,
            "proto_align_metric": "cosine",
            "proto_align_start_epoch": 40,
            "proto_align_full_epoch": 90,
        }
    )
    assert cfg["source_anchor_proto_enabled"] is True
    assert cfg["lambda_proto_align"] == 0.001


def test_train_d_ae_accepts_proto_align_kwargs():
    sig = inspect.signature(pv.train_d_ae)
    for name in (
        "source_anchor_prototypes",
        "lambda_proto_align_eff",
        "proto_align_metric",
        "proto_align_min_count",
        "proto_align_update_source_ema",
    ):
        assert name in sig.parameters


def test_pretrain_main_initializes_source_anchor_prototypes():
    src = inspect.getsource(pv.run_single_experiment)
    assert "resolve_source_anchor_proto_training_params(param)" in src
    assert "source_anchor_proto_enabled" in src
    assert "SourceAnchorEMAPrototypes(" in src
    assert "get_proto_align_lambda_eff(" in src
    assert "source_anchor_prototypes=source_anchor_prototypes" in src
    assert "lambda_proto_align_eff=lambda_proto_align_eff" in src


def test_train_d_ae_adds_proto_align_to_total_loss():
    src = inspect.getsource(pv.train_d_ae)
    assert "proto_align_loss" in src and "lambda_proto_align_eff" in src
    assert "* proto_align_loss" in src


def test_run_single_experiment_logs_proto_metadata():
    src = inspect.getsource(pv.run_single_experiment)
    assert "source_anchor_proto_metrics_payload" in src
    assert "source_anchor_proto" in src
