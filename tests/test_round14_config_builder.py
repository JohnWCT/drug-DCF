#!/usr/bin/env python3
import json
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round14_config_builder import build_round14_pretrain_configs, build_round14_finetune_manifest


@pytest.fixture
def settings_path(tmp_path):
    settings = {
        "round12_root": "result/optimization_runs/round12_proto_alignment",
        "round11_root": "result/optimization_runs/round11_stability_recon",
        "base_routes": [
            {
                "route_id": "exp008_proto_response_route",
                "source_model": "exp_008",
                "source_round": "round12",
                "round14_branch": "14B",
            },
            {
                "route_id": "exp035_strong_zonly_route",
                "source_model": "exp_035",
                "source_round": "round11",
                "round14_branch": "14C",
            },
        ],
        "vicreg": {
            "settings_main": [
                {"lambda_tumor_var": 0.0, "lambda_tumor_cov": 0.0},
                {"lambda_tumor_var": 0.00001, "lambda_tumor_cov": 0.00001},
                {"lambda_tumor_var": 0.00003, "lambda_tumor_cov": 0.00003},
                {"lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0001},
                {"lambda_tumor_var": 0.0003, "lambda_tumor_cov": 0.0003},
                {"lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.00003},
                {"lambda_tumor_var": 0.00003, "lambda_tumor_cov": 0.0001},
                {"lambda_tumor_var": 0.0003, "lambda_tumor_cov": 0.0001},
                {"lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0003},
            ],
            "settings_exp035_small": [
                {"lambda_tumor_var": 0.0, "lambda_tumor_cov": 0.0},
                {"lambda_tumor_var": 0.00001, "lambda_tumor_cov": 0.00001},
                {"lambda_tumor_var": 0.00003, "lambda_tumor_cov": 0.00003},
                {"lambda_tumor_var": 0.0001, "lambda_tumor_cov": 0.0001},
                {"lambda_tumor_var": 0.0003, "lambda_tumor_cov": 0.0003},
            ],
            "schedules": [{"start": 20, "full": 60}, {"start": 40, "full": 90}],
        },
        "seeds": [101, 202, 303],
        "response_feature_modes": ["none", "own_cancer", "own_plus_summary"],
        "finetune": {"config": "config/params_finetune_round14_proto_features.json"},
    }
    path = tmp_path / "round14_settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return str(path)


def test_builds_84_pretrain_jobs(tmp_path, settings_path, monkeypatch):
    fake_params = {
        "conditional_adv_enabled": True,
        "source_anchor_proto_enabled": True,
        "lambda_proto_align": 0.0001,
        "reconstruction_loss_type": "mse",
    }

    def fake_resolve(round_root, exp_id):
        return fake_params.copy(), os.path.join(round_root, "pretrain", exp_id)

    def fake_r11(round11_root, exp_id):
        return fake_params.copy(), os.path.join(round11_root, "pretrain", exp_id)

    monkeypatch.setattr("tools.round14_config_builder._resolve_pretrain_params", fake_resolve)
    monkeypatch.setattr("tools.round14_config_builder._resolve_round11_baseline", fake_r11)

    manifest = build_round14_pretrain_configs(settings_path, str(tmp_path / "round14"), force=True)
    df = pd.read_csv(manifest)
    assert len(df) == 84
    assert set(df["round14_branch"]) == {"14B", "14C"}
    assert (df["lambda_tumor_var"] == 0).any()
    assert (df["lambda_tumor_var"] > 0).any()
    assert df["route_id"].nunique() == 2


def test_build_finetune_manifest_from_selection(tmp_path, settings_path):
    sel = pd.DataFrame(
        [
            {"ID": "exp_001", "result_folder": str(tmp_path / "ckpt1"), "route_id": "exp008_proto_response_route"},
            {"ID": "exp_002", "result_folder": str(tmp_path / "ckpt2"), "route_id": "exp035_strong_zonly_route"},
        ]
    )
    sel_path = tmp_path / "selection.csv"
    sel.to_csv(sel_path, index=False)
    os.makedirs(tmp_path / "ckpt1", exist_ok=True)
    os.makedirs(tmp_path / "ckpt2", exist_ok=True)

    out = build_round14_finetune_manifest(
        settings_path,
        str(tmp_path / "round14"),
        str(sel_path),
        force=True,
    )
    ft = pd.read_csv(out["finetune_dispatch_manifest"])
    assert len(ft) == 2 * 3 * 4
    assert set(ft["prototype_feature_mode"]) == {"none", "own_cancer", "own_plus_summary"}
