#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_pretrain_source_has_vicreg_params_and_mean_metadata():
    text = open(os.path.join(PROJECT_ROOT, "pretrain_VAEwC.py"), encoding="utf-8").read()
    for token in (
        "lambda_tumor_var",
        "lambda_tumor_cov",
        "resolve_tumor_vicreg_training_params",
        "compute_vicreg_var_cov_loss",
        "tumor_vicreg_var_loss_history",
        "tumor_vicreg_var_loss_mean",
        "tumor_vicreg_cov_loss_mean",
        "tumor_vicreg_loss_mean",
    ):
        assert token in text


def test_round14_config_includes_vicreg_lambdas(tmp_path, monkeypatch):
    from tools.round14_config_builder import build_round14_pretrain_configs
    settings = {
        "round12_root": "r12", "round11_root": "r11",
        "base_routes": [
            {"route_id": "exp008_proto_response_route", "source_model": "exp_008", "source_round": "round12", "round14_branch": "14B"},
            {"route_id": "exp035_strong_zonly_route", "source_model": "exp_035", "source_round": "round11", "round14_branch": "14C"},
        ],
        "vicreg": {"settings_main": [{"lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0001}], "settings_exp035_small": [{"lambda_tumor_var": 0.0, "lambda_tumor_cov": 0.0}], "schedules": [{"start": 20, "full": 60}]},
        "seeds": [101],
    }
    settings_path = tmp_path / "s.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    fake = {"conditional_adv_enabled": True, "reconstruction_loss_type": "mse"}
    monkeypatch.setattr("tools.round14_config_builder._resolve_pretrain_params", lambda *_a, **_k: (fake, "/tmp/ckpt"))
    monkeypatch.setattr("tools.round14_config_builder._resolve_round11_baseline", lambda *_a, **_k: (fake, "/tmp/ckpt"))
    manifest = build_round14_pretrain_configs(str(settings_path), str(tmp_path / "out"), force=True)
    df = pd.read_csv(manifest)
    cfg = json.load(open(os.path.join(PROJECT_ROOT, df.iloc[0]["config_path"]), encoding="utf-8"))
    params = cfg["pretrain_param_combinations"][0]
    assert params["lambda_tumor_var"] == 0.0001
    assert params["tumor_vicreg_start_epoch"] == 20


def test_optimization_runner_accepts_round14_selection_mode():
    from tools.optimization_selection import SELECTION_MODES
    assert "round14_vicreg_stabilizer_qc" in SELECTION_MODES
