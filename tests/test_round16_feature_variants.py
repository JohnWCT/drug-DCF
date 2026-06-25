import numpy as np

from tools.prototype_response_features import (
    compute_proto_distance_features,
    parse_feature_variant,
    resolve_feature_mode_options,
)


def _dummy_proto(n=3, d=4):
    z = np.random.randn(d).astype(np.float32)
    src = np.random.randn(n, d).astype(np.float32)
    tgt = np.random.randn(n, d).astype(np.float32)
    mapping = {"id_to_name": {i: f"c{i}" for i in range(n)}}
    return z, 0, src, tgt, mapping


def test_own_plus_summary_no_l2_smaller_dim():
    z, cid, src, tgt, mapping = _dummy_proto()
    full = compute_proto_distance_features(z, cid, src, target_prototypes=tgt, cancer_type_mapping=mapping, mode="own_plus_summary", include_l2_distance=True)
    no_l2 = compute_proto_distance_features(z, cid, src, target_prototypes=tgt, cancer_type_mapping=mapping, mode="own_plus_summary_no_l2")
    assert len(no_l2["feature_names"]) < len(full["feature_names"])
    assert "proto_own_source_l2_dist" not in no_l2["feature_names"]


def test_own_plus_summary_no_gap_removes_gap():
    z, cid, src, tgt, mapping = _dummy_proto()
    out = compute_proto_distance_features(z, cid, src, target_prototypes=tgt, cancer_type_mapping=mapping, mode="own_plus_summary_no_gap", include_same_cancer_gap=False)
    assert "proto_same_cancer_gap" not in out["feature_names"]


def test_own_plus_summary_no_initialized_flags():
    z, cid, src, tgt, mapping = _dummy_proto()
    out = compute_proto_distance_features(
        z, cid, src, target_prototypes=tgt, cancer_type_mapping=mapping,
        mode="own_plus_summary_no_initialized_flags", include_initialized_flag=False,
    )
    assert "proto_source_anchor_initialized" not in out["feature_names"]
    assert "proto_target_proto_initialized" not in out["feature_names"]


def test_robust_scaler_variant_no_nan():
    opts = resolve_feature_mode_options("own_plus_summary_robust_scaler")
    assert opts["proto_feature_scaler"] == "robust"
    z, cid, src, tgt, mapping = _dummy_proto()
    out = compute_proto_distance_features(z, cid, src, target_prototypes=tgt, cancer_type_mapping=mapping, mode="own_plus_summary_robust_scaler")
    assert not np.isnan(out["features"]).any()


def test_feature_names_align_with_matrix_columns():
    z, cid, src, tgt, mapping = _dummy_proto()
    out = compute_proto_distance_features(z, cid, src, target_prototypes=tgt, cancer_type_mapping=mapping, mode="own_plus_summary", include_l2_distance=True)
    assert out["features"].shape[-1] == len(out["feature_names"])
    assert parse_feature_variant("own_plus_summary_zscore")["scaler"] == "standard"
