import numpy as np
import pytest

from tools.prototype_response_features import (
    OWN_PROTO_CONTEXT_MODES,
    build_raw_context_vector,
    compute_own_proto_context_features,
    fit_context_projection,
    get_own_source_target_vectors,
    get_projected_context_dim,
)


def _fixtures(latent_dim=4, n=3):
    z = np.random.randn(latent_dim).astype(np.float32)
    src = np.random.randn(n, latent_dim).astype(np.float32)
    tgt = np.random.randn(n, latent_dim).astype(np.float32)
    mapping = {"id_to_name": {i: f"c{i}" for i in range(n)}}
    return z, 0, src, tgt, mapping


def test_own_proto_delta_feature_dim():
    z, cid, src, tgt, mapping = _fixtures()
    feat, names, meta = compute_own_proto_context_features(
        z, cid, src, target_prototypes=tgt, mode="own_proto_delta", cancer_type_mapping=mapping, strict=False
    )
    latent_dim = z.shape[0]
    summary_dim = 11
    assert len(feat) == latent_dim * 3 + summary_dim
    assert len(names) == len(feat)
    assert meta["feature_dim"] == len(feat)


def test_own_proto_context_feature_dim():
    z, cid, src, tgt, mapping = _fixtures()
    feat, names, _ = compute_own_proto_context_features(
        z, cid, src, target_prototypes=tgt, mode="own_proto_context", cancer_type_mapping=mapping, strict=False
    )
    assert len(feat) == z.shape[0] * 3 + 11
    assert len(names) == len(feat)


def test_projected_16_feature_dim():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8, n=20)
    raw_rows = []
    for i in range(20):
        vecs = get_own_source_target_vectors(i % 3, src, tgt, strict=False, latent_dim=8)
        raw_rows.append(build_raw_context_vector(z, vecs["source_anchor"], vecs["target_proto"]))
    pca = fit_context_projection(np.stack(raw_rows), 16)
    feat, names, meta = compute_own_proto_context_features(
        z,
        cid,
        src,
        target_prototypes=tgt,
        mode="own_proto_context_projected_16",
        cancer_type_mapping=mapping,
        projection_model=pca,
        strict=False,
    )
    assert get_projected_context_dim("own_proto_context_projected_16") == 16
    assert meta["projection_dim"] == 16
    assert len(feat) == int(pca.n_components_) + 11
    assert len(names) == len(feat)
    assert int(pca.n_components_) <= 16


def test_projected_32_feature_dim():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8, n=20)
    raw_rows = []
    for i in range(20):
        vecs = get_own_source_target_vectors(i % 3, src, tgt, strict=False, latent_dim=8)
        raw_rows.append(build_raw_context_vector(z, vecs["source_anchor"], vecs["target_proto"]))
    pca = fit_context_projection(np.stack(raw_rows), 32)
    feat, names, _ = compute_own_proto_context_features(
        z,
        cid,
        src,
        target_prototypes=tgt,
        mode="own_proto_context_projected_32",
        cancer_type_mapping=mapping,
        projection_model=pca,
        strict=False,
    )
    assert len(feat) == int(pca.n_components_) + 11
    assert len(names) == len(feat)
    assert int(pca.n_components_) <= 32


def test_missing_target_strict_raises():
    z, cid, src, _, mapping = _fixtures()
    with pytest.raises(ValueError):
        get_own_source_target_vectors(
            cid,
            src,
            None,
            target_initialized=np.zeros(len(src), dtype=bool),
            strict=True,
            latent_dim=z.shape[0],
        )


def test_strict_false_no_nan():
    z, cid, src, tgt, mapping = _fixtures()
    feat, _, _ = compute_own_proto_context_features(
        z,
        -1,
        src,
        target_prototypes=tgt,
        mode="own_proto_delta",
        cancer_type_mapping=mapping,
        strict=False,
    )
    assert not np.isnan(feat).any()


def test_projection_fit_source_only_domain():
    z, cid, src, tgt, mapping = _fixtures(latent_dim=8, n=5)
    source_rows = []
    for i in range(5):
        vecs = get_own_source_target_vectors(i, src, tgt, strict=False, latent_dim=8)
        source_rows.append(build_raw_context_vector(z, vecs["source_anchor"], vecs["target_proto"]))
    mat = np.stack(source_rows)
    pca = fit_context_projection(mat, 3)
    assert pca.n_components_ == 3
    assert mat.shape[0] == 5


def test_all_own_proto_modes_supported():
    assert "own_proto_delta" in OWN_PROTO_CONTEXT_MODES
    assert "own_proto_interaction" in OWN_PROTO_CONTEXT_MODES
