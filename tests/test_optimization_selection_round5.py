import pandas as pd
from tools.collapse_detection import apply_round5_stage1_filter, rank_round5_stage2
from tools.optimization_selection import select_top_k_with_baselines, SELECTION_MODES


def test_round5_selection_mode_registered():
    assert "round5_structure_first" in SELECTION_MODES


def test_round5_stage1_excludes_collapse():
    df = pd.DataFrame([
        {"ID": "exp_a", "kmeans_ari": 0.70, "wasserstein": 0.4, "fid": 20, "mmd": 0.01, "lambda_proto": 0},
        {"ID": "exp_b", "kmeans_ari": 0.10, "wasserstein": 0.3, "fid": 10, "mmd": 0.01, "lambda_proto": 0},
    ])
    out = apply_round5_stage1_filter(df)
    assert set(out["ID"]) == {"exp_a"}


def test_round5_stage2_orders_wasserstein_first():
    df = pd.DataFrame([
        {"ID": "exp_a", "wasserstein": 0.8, "kmeans_ari": 0.9, "fid": 30, "mmd": 0.02},
        {"ID": "exp_b", "wasserstein": 0.5, "kmeans_ari": 0.7, "fid": 25, "mmd": 0.01},
    ])
    ranked = rank_round5_stage2(df)
    assert ranked.iloc[0]["ID"] == "exp_b"


def test_select_top_k_with_baselines_adds_forced_ids():
    df = pd.DataFrame([
        {"ID": f"exp_{i:03d}", "wasserstein": 0.5 + i * 0.01, "kmeans_ari": 0.7, "fid": 20, "mmd": 0.01, "lambda_proto": 0}
        for i in range(5)
    ])
    selected, info = select_top_k_with_baselines(
        df, df, top_k=2, force_baseline_models=["exp_746", "exp_018"], selection_mode="round5_structure_first"
    )
    ids = set(selected["ID"].astype(str))
    assert "exp_746" in ids
    assert "exp_018" in ids
    assert info["total_selected"] >= 4


def test_merge_result_dir_paths_keeps_primary_and_extends():
    from tools.optimization_selection import merge_result_dir_paths

    merged = merge_result_dir_paths(
        "result/optimization_runs/control/pretrain",
        [
            "result/optimization_runs/class_gap/pretrain",
            "result/optimization_runs/t2s/pretrain",
        ],
    )
    assert len(merged) == 3
    assert merged[0].endswith("control/pretrain")
    assert any(p.endswith("class_gap/pretrain") for p in merged)
    assert any(p.endswith("t2s/pretrain") for p in merged)


def test_round5_selection_info_has_legacy_report_keys():
    import pandas as pd
    from tools.optimization_selection import select_top_k_with_baselines

    df = pd.DataFrame([
        {"ID": "exp_001", "wasserstein": 0.5, "kmeans_ari": 0.7, "fid": 20, "mmd": 0.01, "lambda_proto": 0},
    ])
    _, info = select_top_k_with_baselines(df, df, top_k=1, selection_mode="round5_structure_first")
    for key in ("controls_available", "controls_selected", "ranked_selected", "shortage", "infonce_available"):
        assert key in info
