"""Tests for Round 10 conditional ADV training flags."""

import pytest

from tools.conditional_adv import (
    conditional_adv_metrics_payload,
    get_cond_adv_lambda_eff,
    resolve_conditional_adv_training_params,
)


def test_disabled_preserves_baseline_global_only():
    cfg = resolve_conditional_adv_training_params({"lambda_proto": 0.1})
    assert cfg["conditional_adv_enabled"] is False
    assert cfg["global_adv_mode"] == "baseline_global_only"
    assert cfg["lambda_cond_adv"] == 0.0


def test_conditional_replacement_enables_conditional():
    cfg = resolve_conditional_adv_training_params(
        {
            "conditional_adv_enabled": True,
            "conditional_adv_mode": "cancer_embedding",
            "global_adv_mode": "conditional_replacement",
            "lambda_cond_adv": 0.001,
        }
    )
    assert cfg["conditional_adv_enabled"] is True
    assert cfg["global_adv_mode"] == "conditional_replacement"


def test_weak_global_multiplier_preserved():
    cfg = resolve_conditional_adv_training_params(
        {
            "conditional_adv_enabled": True,
            "conditional_adv_mode": "cancer_embedding",
            "global_adv_mode": "conditional_plus_weak_global",
            "lambda_global_adv_multiplier": 0.25,
            "lambda_cond_adv": 0.001,
        }
    )
    assert cfg["lambda_global_adv_multiplier"] == 0.25


def test_lambda_cond_schedule():
    lam = get_cond_adv_lambda_eff(35, 0.001, 10, 60)
    assert lam == pytest.approx(0.0005)


def test_logging_payload_fields():
    cfg = resolve_conditional_adv_training_params(
        {
            "conditional_adv_enabled": True,
            "conditional_adv_mode": "cancer_embedding",
            "global_adv_mode": "conditional_replacement",
            "round": "round10",
            "round10_branch": "10B_conditional_replacement",
            "source_baseline_exp_id": "exp_048",
            "lambda_cond_adv": 0.001,
        }
    )
    payload = conditional_adv_metrics_payload(
        cfg,
        {
            "lambda_cond_eff": 0.001,
            "cond_critic_loss_mean": 0.5,
            "cond_encoder_adv_loss_mean": -0.2,
            "cond_gp_mean": 0.1,
            "cond_gp_skip_count": 2,
            "cond_gp_pairing_mode": "batch_fallback",
            "num_cancer_types": 18,
        },
    )
    assert payload["round10_branch"] == "10B_conditional_replacement"
    assert payload["conditional_adv_enabled"] is True
    assert payload["cond_gp_pairing_mode"] == "batch_fallback"
