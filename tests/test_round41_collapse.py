import pandas as pd

from tools.collapse_detection import annotate_alignment_collapse, apply_round41_stage1_filter, rank_round41_stage2
from tools.optimization_selection import apply_selection_ranking
from tools.pretrain_proto_schedule import compute_proto_checkpoint_guard


def test_proto_ineffective_guard_flag():
    guard = compute_proto_checkpoint_guard(
        {"lambda_proto": 0.01, "proto_start_epoch": 10, "proto_full_epoch": 30},
        best_gan_epoch_overall=5,
        best_gan_epoch_post_proto=0,
    )
    assert guard["proto_not_effective_checkpoint"] is True
    assert guard["proto_effective_checkpoint_available"] is False
    assert guard["proto_invalid"] is True


def test_collapse_marks_exp142_like():
    df = pd.DataFrame([{"ID": "exp_142", "kmeans_ari": 0.05, "wasserstein": 0.37, "lambda_proto": 0.01}])
    out = annotate_alignment_collapse(df)
    assert bool(out.iloc[0]["alignment_collapse"]) is True
    assert out.iloc[0]["collapse_reason"] == "global_alignment_destroyed_tumor_structure"


def test_round41_selection_prefers_structure_over_wasserstein():
    df = pd.DataFrame(
        [
            {"ID": "A", "wasserstein": 0.37, "kmeans_ari": 0.05, "fid": 10, "mmd": 0.1, "lambda_proto": 0.01},
            {"ID": "B", "wasserstein": 0.57, "kmeans_ari": 0.74, "fid": 20, "mmd": 0.05, "lambda_proto": 0.01},
        ]
    )
    filtered = apply_round41_stage1_filter(df)
    assert "A" not in filtered["ID"].values
    assert "B" in filtered["ID"].values
    ranked = rank_round41_stage2(filtered)
    assert ranked.iloc[0]["ID"] == "B"


def test_round41_selection_mode_same_as_rank():
    df = pd.DataFrame(
        [
            {"ID": "A", "wasserstein": 0.37, "kmeans_ari": 0.05, "fid": 10, "mmd": 0.1, "lambda_proto": 0.01},
            {"ID": "B", "wasserstein": 0.57, "kmeans_ari": 0.74, "fid": 20, "mmd": 0.05, "lambda_proto": 0.01},
            {"ID": "C", "wasserstein": 0.55, "kmeans_ari": 0.70, "fid": 18, "mmd": 0.04, "lambda_proto": 0.0},
        ]
    )
    pool = apply_round41_stage1_filter(df[df["lambda_proto"] > 0])
    ranked = apply_selection_ranking(pool, selection_mode="round4_1_structure_first")
    assert ranked.iloc[0]["ID"] == "B"
