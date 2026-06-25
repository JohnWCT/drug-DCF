import json

import pandas as pd
import pytest

from tools.round16_bruteforce_config_builder import build_round16_configs


@pytest.fixture
def settings_path(tmp_path):
    settings = {
        "round12_root": "result/optimization_runs/round12_proto_alignment",
        "round11_root": "result/optimization_runs/round11_stability_recon",
        "round15_root": "result/optimization_runs/round15_repro_rescue",
        "stage16e_own_proto_context": {
            "models": ["r13_exp_008", "r15c_exp_005", "r15c_exp_024"],
            "feature_modes": [
                "none",
                "own_plus_summary",
                "own_proto_delta",
                "own_proto_context",
                "own_proto_context_projected_16",
                "own_proto_context_projected_32",
            ],
            "max_combos": 8,
            "seeds": [101, 202, 303],
        },
        "finetune": {"config": "config/params_finetune_round16_bruteforce.json"},
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return str(path)


def test_stage16e_manifest_models_and_projection_flags(settings_path, tmp_path):
    try:
        out = build_round16_configs(settings_path, str(tmp_path / "out"), stage="16e", force=True)
    except FileNotFoundError:
        pytest.skip("pretrain checkpoints not available")
    proto = pd.read_csv(out["stage16e_proto_feature_manifest"])
    fin = pd.read_csv(out["stage16e_finetune_dispatch_manifest"])
    for model in ("r13_exp_008", "r15c_exp_005", "r15c_exp_024"):
        assert proto["model_id"].astype(str).str.contains(model).any()
    assert "own_proto_delta" in set(proto["feature_mode"])
    assert "own_proto_context_projected_16" in set(proto["feature_mode"])
    proj = proto[proto["feature_mode"] == "own_proto_context_projected_16"].iloc[0]
    assert bool(proj["requires_projection"]) is True
    assert int(proj["projection_dim"]) == 16
    assert str(proj["projection_fit_domain"]) == "source_only"
    assert (fin["stage"] == "16e").all()
    assert len(fin) == 3 * 6 * 8 * 3
