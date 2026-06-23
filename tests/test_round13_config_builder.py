#!/usr/bin/env python3
import json
import os
import sys
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round13_config_builder import build_round13_configs, _parse_round12_best_from_docs


def test_builds_manifests_with_mock_pool(tmp_path, monkeypatch):
    settings = {
        "round12_root": "result/optimization_runs/round12_proto_alignment",
        "round11_root": "result/optimization_runs/round11_stability_recon",
        "round12_final_report": "docs/round12_final_report.md",
        "model_pool": {"primary_best_model": "exp_037", "include_round11_exp035": False, "max_models": 1},
        "feature_modes": ["none", "own_cancer"],
        "optional_feature_modes": [],
        "finetune": {"config": "config/params_finetune_proto_features.json"},
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    os.makedirs(tmp_path / "exp_037", exist_ok=True)

    def fake_resolve(_settings):
        return [
            {
                "source_model_id": "exp_037",
                "role": "round12_best_downstream",
                "source_round": "round12",
                "checkpoint_dir": str(tmp_path / "exp_037"),
            }
        ], []

    monkeypatch.setattr("tools.round13_config_builder.resolve_model_pool", fake_resolve)
    out = build_round13_configs(str(settings_path), str(tmp_path / "round13"), force=True)
    manifest = pd.read_csv(out["finetune_dispatch_manifest"])
    assert (manifest["prototype_feature_mode"] == "none").any()
    assert manifest["model_select_path"].notna().all()


def test_parses_exp037_from_docs():
    best = _parse_round12_best_from_docs(os.path.join(PROJECT_ROOT, "docs/round12_final_report.md"))
    assert best == "exp_037"
