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


def test_load_and_enrich_skips_empty_branch(monkeypatch):
    import tools.optimization_selection as sel

    calls = []
    monkeypatch.setattr(
        sel,
        "load_all_pretrain_rows",
        lambda d, source_tag="": (
            __import__("pandas").DataFrame()
            if "empty" in d
            else __import__("pandas").DataFrame([{"ID": "exp_ok", "kmeans_ari": 0.7, "wasserstein": 0.5}])
        ),
    )
    monkeypatch.setattr(
        sel,
        "enrich_selection_metadata",
        lambda df, rd: (calls.append(rd), df.assign(_enriched_from=rd))[1],
    )
    out = sel.load_and_enrich_merged_results(["/tmp/control", "/tmp/empty", "/tmp/class_gap"])
    assert len(out) == 2
    assert len(calls) == 2


def test_round5_multi_result_selection_does_not_reenrich_with_primary_only(monkeypatch, tmp_path):
    import os
    import pandas as pd
    import tools.optimization_selection as sel

    control_dir = tmp_path / "control"
    gap_dir = tmp_path / "class_gap"
    control_dir.mkdir()
    gap_dir.mkdir()
    run_dir = tmp_path / "run"

    enrich_calls = []

    def fake_load(result_dir, source_tag=""):
        tag = source_tag or os.path.basename(result_dir)
        return pd.DataFrame(
            [
                {
                    "ID": f"exp_{tag}",
                    "kmeans_ari": 0.7,
                    "wasserstein": 0.5,
                    "fid": 20.0,
                    "mmd": 0.01,
                    "lambda_proto": 0.0,
                    "pretrain_run_tag": tag,
                }
            ]
        )

    def tracking_enrich(df, result_dir):
        enrich_calls.append(result_dir)
        out = df.copy()
        if "class_gap" in result_dir:
            out["latent_size"] = 64
            out["lambda_class_gap"] = 0.001
        else:
            out["latent_size"] = 32
            out["lambda_class_gap"] = 0.0
        from tools.collapse_detection import annotate_alignment_collapse

        return annotate_alignment_collapse(out)

    monkeypatch.setattr(sel, "load_all_pretrain_rows", fake_load)
    monkeypatch.setattr(sel, "enrich_selection_metadata", tracking_enrich)

    sel.write_selection_outputs(
        str(run_dir),
        str(control_dir),
        result_dirs=[str(gap_dir)],
        selection_mode="round5_structure_first",
        no_filter=True,
        min_passing=1,
        require_controls=0,
        top_k=5,
    )

    assert len(enrich_calls) == 2
    filtered = pd.read_csv(run_dir / "selection" / "pretrain_filtered_candidates.csv")
    gap_row = filtered[filtered["pretrain_run_tag"] == "class_gap"].iloc[0]
    assert int(gap_row["latent_size"]) == 64
    assert float(gap_row["lambda_class_gap"]) == 0.001
