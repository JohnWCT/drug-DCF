"""Verify pretrain_VAEwC.py is wired for Round 10 conditional ADV."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.conditional_adv import build_conditional_adv_components


ROUND10_10B_PARAM = {
    "round": "round10",
    "round10_branch": "10B_conditional_replacement",
    "source_baseline_exp_id": "exp_048",
    "conditional_adv_enabled": True,
    "conditional_adv_mode": "cancer_embedding",
    "cancer_condition_dim": 8,
    "lambda_cond_adv": 0.0001,
    "cond_adv_start_epoch": 10,
    "cond_adv_full_epoch": 40,
    "global_adv_mode": "conditional_replacement",
    "lambda_global_adv_multiplier": 0.0,
    "gan_learning_rate": 0.0005,
    "train_num_epochs": 600,
}


def test_pretrain_module_has_conditional_adv_symbols():
    pretrain = importlib.import_module("pretrain_VAEwC")
    source = inspect.getsource(pretrain)
    required = (
        "train_cond_discrim",
        "build_conditional_adv_components",
        "conditional_adv_metrics_payload",
        "global_adv_mode",
        "lambda_cond_eff",
        "cond_critic",
    )
    for symbol in required:
        assert symbol in source, f"pretrain_VAEwC.py missing {symbol}"


def test_build_conditional_adv_components_disabled():
    bundle = build_conditional_adv_components(
        {},
        latent_size=64,
        num_cancer_types=18,
        gan_learning_rate=5e-4,
        gan_epoch=10,
        device=torch.device("cpu"),
    )
    assert bundle["cond_cfg"]["conditional_adv_enabled"] is False
    assert bundle["cond_critic"] is None
    assert bundle["cond_critic_optimizer"] is None


def test_build_conditional_adv_components_enabled_10b():
    bundle = build_conditional_adv_components(
        ROUND10_10B_PARAM,
        latent_size=64,
        num_cancer_types=18,
        gan_learning_rate=5e-4,
        gan_epoch=10,
        device=torch.device("cpu"),
    )
    cfg = bundle["cond_cfg"]
    assert cfg["conditional_adv_enabled"] is True
    assert cfg["global_adv_mode"] == "conditional_replacement"
    assert cfg["lambda_cond_adv"] == 0.0001
    assert bundle["cond_critic"] is not None
    assert bundle["cond_critic_optimizer"] is not None
    assert bundle["cond_critic_scheduler"] is not None

    z = torch.randn(4, 64)
    labels = torch.tensor([0, 1, 2, 3])
    scores = bundle["cond_critic"](z, labels)
    assert scores.shape == (4,)


@pytest.mark.skipif(
    os.environ.get("RUN_ROUND10_GPU_SMOKE") != "1",
    reason="Set RUN_ROUND10_GPU_SMOKE=1 to run GPU smoke artifact check",
)
def test_round10_smoke_gan_metrics_contain_conditional_adv():
    smoke_dir = os.path.join(
        PROJECT_ROOT,
        "result/optimization_runs/round10_cond_adv_pretrain_smoke_direct/exp_001",
    )
    gan_path = os.path.join(smoke_dir, "gan_metrics.json")
    if not os.path.exists(gan_path):
        pytest.skip(f"Smoke artifacts missing: {gan_path}")

    metrics = json.load(open(gan_path, encoding="utf-8"))
    assert metrics.get("conditional_adv_enabled") is True
    assert metrics.get("round10_branch") == "10B_conditional_replacement"
    assert metrics.get("global_adv_mode") == "conditional_replacement"
    assert float(metrics.get("lambda_cond_adv", 0)) > 0
    assert metrics.get("cond_critic_loss_mean") is not None
    assert metrics.get("cond_encoder_adv_loss_mean") is not None
