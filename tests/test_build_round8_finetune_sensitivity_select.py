import pandas as pd

from tools.build_round8_finetune_sensitivity_select import build_sensitivity_select


def _aggregate_df():
    return pd.DataFrame(
        {
            "Model_ID": ["exp_a", "exp_b", "exp_c", "exp_048"],
            "Average_TCGA_AUC_mean": [0.60, 0.58, 0.57, 0.55],
            "Global_TCGA_AUC_mean": [0.59, 0.57, 0.56, 0.54],
        }
    )


def _selection_df():
    return pd.DataFrame(
        {
            "ID": ["exp_a", "exp_b", "exp_c", "exp_048", "exp_021"],
            "round8_vicreg_active": [True, True, False, True, True],
            "round8_control_like": [False, False, True, True, False],
            "pretrain_result_dir": ["pretrain/exp_a"] * 5,
        }
    )


def test_build_sensitivity_select_top_and_force():
    out = build_sensitivity_select(
        _aggregate_df(),
        _selection_df(),
        force_models=["exp_048", "exp_021", "exp_746"],
        max_models=12,
    )
    ids = out["ID"].astype(str).tolist()
    assert ids[0] == "exp_a"
    assert "exp_048" in ids
    assert "exp_021" in ids
    assert "exp_746" in ids
    assert len(ids) == len(set(ids))
    assert "selection_rank" in out.columns


def test_max_models_limits_output():
    out = build_sensitivity_select(
        _aggregate_df(),
        _selection_df(),
        force_models=["exp_048"],
        max_models=3,
    )
    assert len(out) == 3
