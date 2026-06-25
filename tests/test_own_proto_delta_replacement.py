import numpy as np
import pytest

from tools.prototype_response_features import (
    compute_own_proto_delta_replacement_features,
    compute_own_proto_delta_vectors,
    fit_context_projection,
    get_projected_delta_dim,
)


def _fixtures(latent_dim=8, n=3):
    z = np.random.randn(latent_dim).astype(np.float32)
    src = np.random.randn(n, latent_dim).astype(np.float32)
    tgt = np.random.randn(n, latent_dim).astype(np.float32)
    mapping = {"id_to_name": {i: f"c{i}" for i in range(n)}}
    return z, 0, src, tgt, mapping


def test_own_proto_delta_only_dim():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8)
    feat, names, meta = compute_own_proto_delta_replacement_features(
        z, cid, src, target_prototypes=tgt, mode="own_proto_delta_only", cancer_type_mapping=mapping, strict=False
    )
    assert len(feat) == 3 * 8
    assert len(names) == len(feat)
    assert meta["uses_delta"] is True
    assert meta["uses_own_plus_summary"] is False


def test_own_plus_summary_plus_delta_dim():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8)
    feat, names, meta = compute_own_proto_delta_replacement_features(
        z, cid, src, target_prototypes=tgt, mode="own_plus_summary_plus_delta", cancer_type_mapping=mapping, strict=False
    )
    summary_dim = 11
    assert len(feat) == summary_dim + 3 * 8
    assert len(names) == len(feat)
    assert meta["uses_delta"] and meta["uses_own_plus_summary"]


def test_no_delta_control_matches_summary():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8)
    summary, _, _ = compute_own_proto_delta_replacement_features(
        z, cid, src, target_prototypes=tgt, mode="own_plus_summary", cancer_type_mapping=mapping, strict=False
    )
    control, _, meta = compute_own_proto_delta_replacement_features(
        z, cid, src, target_prototypes=tgt, mode="own_plus_summary_no_delta_control", cancer_type_mapping=mapping, strict=False
    )
    assert np.allclose(summary, control)
    assert meta["mode"] == "own_plus_summary_no_delta_control"


def test_projected_delta_dims():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8, n=20)
    delta, _ = compute_own_proto_delta_vectors(z, src[cid], tgt[cid])
    raw = np.stack([delta] * 20)
    pca = fit_context_projection(raw, 16)
    feat, names, meta = compute_own_proto_delta_replacement_features(
        z, cid, src, target_prototypes=tgt, mode="own_proto_delta_projected_16",
        cancer_type_mapping=mapping, projection_model=pca, strict=False,
    )
    assert len(feat) == int(pca.n_components_)
    assert meta["uses_projection"] is True
    assert get_projected_delta_dim("own_proto_delta_projected_16") == 16


def test_strict_false_no_nan():
    z, _, src, tgt, mapping = _fixtures()
    feat, _, _ = compute_own_proto_delta_replacement_features(
        z, -1, src, target_prototypes=tgt, mode="own_proto_delta_only", cancer_type_mapping=mapping, strict=False
    )
    assert not np.isnan(feat).any()
