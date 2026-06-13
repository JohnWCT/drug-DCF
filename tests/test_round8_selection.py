import pandas as pd

from tools.optimization_selection import DEFAULT_FORCE_BASELINE_PATHS, SELECTION_MODES
from tools.round8_selection import (
    annotate_round8_scores,
    encoder_family,
    is_vicreg_active,
    select_round8_architecture_broad_probe,
)


def _pool_df():
    return pd.DataFrame(
        {
            "ID": ["vicreg_a", "control_a", "collapse_a", "lat96_a", "wide_a", "kmeans_a"],
            "kmeans_ari": [0.55, 0.60, 0.20, 0.58, 0.52, 0.75],
            "wasserstein": [0.55, 0.60, 0.50, 0.62, 0.58, 0.65],
            "latent_size": [64, 64, 64, 96, 64, 64],
            "encoder_dims": [[512, 256, 128]] * 5 + [[768, 384, 192]],
            "dropout_rate": [0.05, 0.10, 0.10, 0.05, 0.05, 0.10],
            "alignment_collapse": [False, False, True, False, False, False],
            "structure_pass": [True, True, False, True, True, True],
            "lambda_proto": [0, 0, 0, 0, 0, 0],
            "lambda_tumor_topology": [0, 0, 0, 0, 0, 0],
            "lambda_class_gap": [0, 0, 0, 0, 0, 0],
            "lambda_tumor_supcon": [0, 0, 0, 0, 0, 0],
            "lambda_subspace_ortho": [0, 0, 0, 0, 0, 0],
            "lambda_tumor_var": [0.0003, 0, 0, 0.0002, 0.0003, 0],
            "lambda_tumor_cov": [0.0003, 0, 0, 0.0002, 0.0003, 0],
            "fid": [20, 22, 25, 21, 23, 18],
            "mmd": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )


def test_round8_selection_mode_registered():
    assert "round8_architecture_broad_probe" in SELECTION_MODES


def test_force_baseline_paths_include_exp048_exp021():
    assert "exp_048" in DEFAULT_FORCE_BASELINE_PATHS
    assert "exp_021" in DEFAULT_FORCE_BASELINE_PATHS


def test_vicreg_active_group():
    row = pd.Series({"lambda_tumor_var": 0.0003, "lambda_tumor_cov": 0.0003, "lambda_tumor_topology": 0})
    assert is_vicreg_active(row)


def test_encoder_family_labels():
    row = pd.Series({"encoder_dims": [768, 384, 192]})
    assert encoder_family(row) == "wide_768"


def test_diverse_selection_keeps_forced_baselines_and_groups():
    pool = annotate_round8_scores(_pool_df())
    selected, info = select_round8_architecture_broad_probe(
        pool,
        pool,
        top_k=20,
        force_baseline_models=["exp_048", "exp_021", "exp_746"],
    )
    assert "round8_selection_group" in selected.columns
    assert "round8_diversity_reason" in selected.columns
    forced = selected[selected["round8_selection_group"] == "G9_forced_baseline"]["ID"].astype(str).tolist()
    assert "exp_048" in forced
    assert "exp_021" in forced
    assert "exp_746" in forced
    assert "collapse_a" not in set(selected["ID"].astype(str))
    assert info["selection_mode"] == "round8_architecture_broad_probe"


def test_architecture_diversity_groups():
    pool = annotate_round8_scores(_pool_df())
    selected, _ = select_round8_architecture_broad_probe(pool, pool, top_k=12)
    groups = set(selected["round8_selection_group"])
    assert "G1_vicreg_active_best" in groups or selected["round8_vicreg_active"].fillna(False).any()
    assert selected["round8_control_like"].fillna(False).any()


def test_runner_parser_accepts_round8_selection_mode():
    from tools.optimization_runner import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "select",
        "--run-dir", "dummy_run",
        "--result-dir", "dummy_result",
        "--selection-mode", "round8_architecture_broad_probe",
    ])
    assert args.selection_mode == "round8_architecture_broad_probe"
