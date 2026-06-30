"""Round 17 direct-prototype feature modes."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from tools.prototype_response_features import (
    ROUND17_STANDALONE_MODES,
    SUPPORTED_MODES,
    build_projection_raw_row,
    compute_own_proto_delta_replacement_features,
    compute_round17_standalone_features,
    fit_context_projection,
    get_projected_delta_dim,
    is_round17_standalone_mode,
)

ROUND17_NEW_MODES = (
    "own_proto_delta_projected_8",
    "own_proto_delta_projected_64",
    "own_plus_summary_plus_delta_projected_16",
    "source_proto_delta_projected_16",
    "target_available_context_projected_16",
    "minimal_source_only_min_margin",
)


@pytest.mark.parametrize("mode", ROUND17_NEW_MODES)
def test_round17_new_modes_supported(mode):
    assert mode in SUPPORTED_MODES


def test_projected_delta_dims_include_8_and_64():
    assert get_projected_delta_dim("own_proto_delta_projected_8") == 8
    assert get_projected_delta_dim("own_proto_delta_projected_64") == 64
    assert get_projected_delta_dim("source_proto_delta_projected_16") == 16


def _proto_fixtures(latent_dim: int = 32, n_cancers: int = 3):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(n_cancers, latent_dim)).astype(np.float32)
    target = source + 0.1
    z = rng.normal(size=(latent_dim,)).astype(np.float32)
    mapping = {"id_to_name": {i: f"c{i}" for i in range(n_cancers)}}
    return source, target, z, mapping


def test_source_proto_delta_projected_16_outputs_16_dims():
    source, target, z, mapping = _proto_fixtures()
    raw = np.stack(
        [build_projection_raw_row(z, source[0], target[0], "source_proto_delta_projected_16") for _ in range(20)],
        axis=0,
    )
    pca = fit_context_projection(raw, 16)
    feat, names, meta = compute_own_proto_delta_replacement_features(
        z,
        0,
        source,
        target,
        mode="source_proto_delta_projected_16",
        cancer_type_mapping=mapping,
        projection_model=pca,
        include_initialized_flag=False,
        strict=False,
    )
    assert feat.shape[0] == 16
    assert len(names) == 16
    assert meta["uses_projection"] is True


def test_minimal_source_only_min_margin_has_three_features():
    source, target, z, mapping = _proto_fixtures()
    feat, names, meta = compute_round17_standalone_features(
        z,
        0,
        source,
        target,
        mode="minimal_source_only_min_margin",
        cancer_type_mapping=mapping,
        include_initialized_flag=False,
        strict=False,
    )
    assert feat.shape[0] == 3
    assert names == [
        "proto_own_source_cosine_dist",
        "proto_source_min_dist",
        "proto_source_top1_margin",
    ]
    assert is_round17_standalone_mode("minimal_source_only_min_margin")
    assert "minimal_source_only_min_margin" in ROUND17_STANDALONE_MODES
