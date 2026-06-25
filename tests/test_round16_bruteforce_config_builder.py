import json
import os
import tempfile

import pandas as pd
import pytest

from tools.round16_bruteforce_config_builder import build_round16_configs


@pytest.fixture
def smoke_settings(tmp_path):
    settings = {
        "round12_root": "result/optimization_runs/round12_proto_alignment",
        "round11_root": "result/optimization_runs/round11_stability_recon",
        "round15_root": "result/optimization_runs/round15_repro_rescue",
        "stage16a": {
            "models": ["r13_exp_008", "r15c_exp_005", "r15c_exp_024", "r13_exp_035"],
            "feature_modes": ["none", "own_plus_summary", "own_plus_summary_no_l2", "own_plus_summary_robust_scaler"],
            "seeds": [101, 202, 303],
            "max_combos": 24,
        },
        "finetune": {"config": "config/params_finetune_round16_bruteforce.json", "epochs": 1500},
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return str(path)


def test_stage16a_manifest_models_and_job_count(smoke_settings, tmp_path):
    try:
        out = build_round16_configs(smoke_settings, str(tmp_path / "out"), stage="16a", force=True)
    except FileNotFoundError:
        pytest.skip("pretrain checkpoints not available in this environment")
    manifest = pd.read_csv(out["finetune_dispatch_manifest"])
    for model in ("r13_exp_008", "r15c_exp_005", "r15c_exp_024", "r13_exp_035"):
        assert manifest["model_id"].astype(str).str.contains(model).any(), model
    assert set(manifest["feature_mode"]) >= {"none", "own_plus_summary", "own_plus_summary_no_l2", "own_plus_summary_robust_scaler"}
    assert set(manifest["seed"].unique()) == {101, 202, 303}
    expected = 4 * 4 * 24 * 3
    assert len(manifest) == expected
