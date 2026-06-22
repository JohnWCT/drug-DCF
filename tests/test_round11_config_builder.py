"""Tests for Round 11 config builder."""

import json
import os
import tempfile

import pandas as pd

from tools.round11_config_builder import build_round11_configs


def _fake_baseline_dir(tmp: str) -> str:
    ckpt = os.path.join(tmp, "exp_048")
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, "params.json"), "w") as f:
        json.dump(
            {
                "params": {
                    "latent_size": 64,
                    "encoder_dims": [512, 256, 128],
                    "dropout_rate": 0.1,
                    "pretrain_num_epochs": 10,
                    "pretrain_learning_rate": 0.001,
                    "lambda_cls": 20,
                    "lambda_adv": 1.0,
                }
            },
            f,
        )
    resolved = os.path.join(tmp, "resolved_baselines.csv")
    pd.DataFrame(
        [{"exp_id": "exp_048", "checkpoint_dir": ckpt, "resolved": True}]
    ).to_csv(resolved, index=False)
    return resolved


def test_build_round11_configs_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        resolved = _fake_baseline_dir(tmp)
        settings = {
            "round": "round11",
            "purpose": "test",
            "primary_baseline_exp_id": "exp_048",
            "resolved_baselines": resolved,
            "seeds": [101],
            "conditional_adv": {"mode": "cancer_embedding", "hidden_dims": [64], "dropout": 0.1},
            "round11b_condadv_stabilization": {
                "enabled": True,
                "lambda_cond_adv": [0.0001],
                "schedules": [{"start": 10, "full": 60}],
                "lambda_global_adv_multiplier": [0.25],
                "condition_dims": [16],
                "include_small_10b_control": False,
            },
            "round11c_reconstruction_ablation": {
                "enabled": True,
                "reconstruction_loss_types": ["mse", "smooth_l1"],
                "smooth_l1_beta": [0.5],
                "hybrid_reconstruction_alpha": [0.5],
            },
        }
        settings_path = os.path.join(tmp, "round11_settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings, f)
        outdir = os.path.join(tmp, "round11")
        manifest = build_round11_configs(settings_path, outdir, force=True)
        df = pd.read_csv(manifest)
        assert len(df) >= 3
        assert "reconstruction_loss_type" in df.columns
        assert "round11_branch" in df.columns
        smooth = df[df["reconstruction_loss_type"] == "smooth_l1"]
        assert not smooth.empty
        assert (smooth["smooth_l1_beta"] == 0.5).any()
        ten_c = df[df["global_adv_mode"] == "conditional_plus_weak_global"]
        assert not ten_c.empty
        cfg_dir = os.path.join(outdir, "configs")
        cfg = json.load(open(os.path.join(cfg_dir, os.listdir(cfg_dir)[0])))
        params = cfg["pretrain_param_combinations"][0]
        assert params.get("lambda_proto", 1) == 0
