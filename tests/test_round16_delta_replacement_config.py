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
        "stage16f_delta_replacement": {
            "models": ["r13_exp_008", "r15c_exp_005", "r15c_exp_024", "r13_exp_035"],
            "feature_modes": ["none", "own_plus_summary", "own_proto_delta_only", "own_plus_summary_plus_delta"],
            "max_combos": 8,
            "seeds": [101, 202, 303],
            "finetune_config": "config/params_finetune_round16_delta_replacement.json",
        },
        "finetune": {"config": "config/params_finetune_round16_bruteforce.json"},
    }
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return str(p)

def test_stage16f_manifest(settings_path, tmp_path):
    try:
        out = build_round16_configs(settings_path, str(tmp_path / "out"), stage="16f", force=True)
    except FileNotFoundError:
        pytest.skip("checkpoints missing")
    proto = pd.read_csv(out["stage16f_proto_feature_manifest"])
    fin = pd.read_csv(out["stage16f_finetune_dispatch_manifest.csv"] if False else out["stage16f_finetune_dispatch_manifest"])
    for mode in ("none", "own_plus_summary", "own_proto_delta_only", "own_plus_summary_plus_delta"):
        assert mode in set(proto["feature_mode"])
    for model in ("r13_exp_008", "r15c_exp_005", "r15c_exp_024", "r13_exp_035"):
        assert proto["model_id"].astype(str).str.contains(model).any()
    assert len(fin) == 4 * 4 * 8 * 3
    delta_only = proto[proto["feature_mode"] == "own_proto_delta_only"].iloc[0]
    assert bool(delta_only["uses_delta"]) is True
    assert bool(delta_only["uses_own_plus_summary"]) is False
