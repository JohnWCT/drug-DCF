"""Tests for tools/conditional_adv.py."""

import pytest
import torch

from tools.conditional_adv import (
    CancerConditionEncoder,
    ConditionalDomainCritic,
    compute_conditional_gradient_penalty,
    get_cond_adv_lambda_eff,
    resolve_conditional_adv_training_params,
)


def test_cancer_condition_encoder_shape():
    enc = CancerConditionEncoder(num_cancer_types=5, condition_dim=16)
    cancer_type = torch.tensor([0, 1, 2, 3])
    out = enc(cancer_type)
    assert out.shape == (4, 16)


@pytest.mark.parametrize("condition_dim", [8, 16, 32])
def test_conditional_domain_critic_output_shape(condition_dim):
    critic = ConditionalDomainCritic(
        latent_size=64,
        num_cancer_types=10,
        condition_dim=condition_dim,
    )
    z = torch.randn(6, 64)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    scores = critic(z, labels)
    assert scores.shape == (6,)


def test_invalid_cancer_id_fail_fast():
    enc = CancerConditionEncoder(num_cancer_types=3, condition_dim=8)
    with pytest.raises(ValueError, match="Invalid cancer_type id"):
        enc(torch.tensor([3]))


def test_conditional_gradient_penalty_backward():
    critic = ConditionalDomainCritic(latent_size=8, num_cancer_types=4, condition_dim=4)
    real_z = torch.randn(4, 8, requires_grad=True)
    fake_z = torch.randn(4, 8)
    cancer = torch.tensor([0, 1, 2, 3])
    gp = compute_conditional_gradient_penalty(
        critic, real_z, fake_z, cancer, device=torch.device("cpu"), gp_weight=10.0
    )
    gp.backward()
    assert gp.ndim == 0


@pytest.mark.parametrize(
    "epoch,start,full,lam,expected",
    [
        (5, 10, 60, 0.001, 0.0),
        (10, 10, 60, 0.001, 0.0),
        (35, 10, 60, 0.001, 0.0005),
        (60, 10, 60, 0.001, 0.001),
        (80, 10, 60, 0.001, 0.001),
        (25, 20, 20, 0.002, 0.002),
    ],
)
def test_get_cond_adv_lambda_eff_ramp(epoch, start, full, lam, expected):
    got = get_cond_adv_lambda_eff(epoch, lam, start, full)
    assert got == pytest.approx(expected)


def test_resolve_disabled_backward_compat():
    cfg = resolve_conditional_adv_training_params({})
    assert cfg["conditional_adv_enabled"] is False
    assert cfg["global_adv_mode"] == "baseline_global_only"
    assert cfg["lambda_cond_adv"] == 0.0
