#!/usr/bin/env python3
"""Tests for tools.prototype_response_features."""

from __future__ import annotations

import numpy as np
import pytest

from tools.prototype_response_features import (
    SENTINEL_DISTANCE,
    compute_proto_distance_features,
    concat_latent_and_proto_features,
)


@pytest.fixture
def proto_setup():
    mapping = {
        "id_to_name": {0: "Brain", 1: "Lung", 2: "Ovarian"},
        "name_to_id": {"Brain": 0, "Lung": 1, "Ovarian": 2},
        "cancer_names": ["Brain", "Lung", "Ovarian"],
    }
    dim = 4
    source = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = source + 0.1
    init = np.array([True, True, False], dtype=bool)
    return mapping, source, target, init, dim


def test_mode_none_returns_zero_columns():
    pack = compute_proto_distance_features(
        np.zeros(4),
        0,
        np.zeros((3, 4)),
        mode="none",
    )
    assert pack["features"].shape == (0,)
    assert pack["feature_names"] == []


def test_own_cancer_feature_shape(proto_setup):
    mapping, source, target, init, dim = proto_setup
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pack = compute_proto_distance_features(
        z,
        0,
        source,
        target_prototypes=target,
        cancer_type_mapping=mapping,
        mode="own_cancer",
        include_l2_distance=True,
        source_initialized=init,
        target_initialized=init,
    )
    assert pack["features"].shape == (7,)
    assert len(pack["feature_names"]) == 7


def test_all_source_anchors_shape(proto_setup):
    mapping, source, target, init, dim = proto_setup
    z = np.ones(dim, dtype=np.float32)
    pack = compute_proto_distance_features(
        z,
        1,
        source,
        cancer_type_mapping=mapping,
        mode="all_source_anchors",
        source_initialized=init,
    )
    assert pack["features"].shape == (3,)


def test_all_source_and_target_shape(proto_setup):
    mapping, source, target, init, dim = proto_setup
    z = np.ones(dim, dtype=np.float32)
    pack = compute_proto_distance_features(
        z,
        1,
        source,
        target_prototypes=target,
        cancer_type_mapping=mapping,
        mode="all_source_and_target",
        source_initialized=init,
        target_initialized=init,
    )
    assert pack["features"].shape == (6,)


def test_own_plus_summary_shape(proto_setup):
    mapping, source, target, init, dim = proto_setup
    z = np.ones(dim, dtype=np.float32)
    pack = compute_proto_distance_features(
        z,
        1,
        source,
        target_prototypes=target,
        cancer_type_mapping=mapping,
        mode="own_plus_summary",
        include_l2_distance=True,
        source_initialized=init,
        target_initialized=init,
    )
    assert pack["features"].shape == (11,)


def test_cosine_distance_no_nan(proto_setup):
    mapping, source, target, init, dim = proto_setup
    z = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pack = compute_proto_distance_features(
        z,
        0,
        source,
        cancer_type_mapping=mapping,
        mode="own_cancer",
        source_initialized=init,
    )
    assert np.all(np.isfinite(pack["features"]))


def test_strict_missing_prototype_raises(proto_setup):
    mapping, source, target, init, dim = proto_setup
    with pytest.raises(ValueError):
        compute_proto_distance_features(
            np.ones(dim),
            99,
            source,
            cancer_type_mapping=mapping,
            mode="own_cancer",
            strict=True,
            source_initialized=init,
        )


def test_strict_false_uses_sentinel(proto_setup):
    mapping, source, target, init, dim = proto_setup
    pack = compute_proto_distance_features(
        np.ones(dim),
        99,
        source,
        cancer_type_mapping=mapping,
        mode="own_cancer",
        strict=False,
        source_initialized=init,
    )
    assert float(pack["features"][0]) == SENTINEL_DISTANCE


def test_feature_names_match_columns(proto_setup):
    mapping, source, target, init, dim = proto_setup
    pack = compute_proto_distance_features(
        np.ones(dim),
        1,
        source,
        cancer_type_mapping=mapping,
        mode="all_source_anchors",
        source_initialized=init,
    )
    assert len(pack["feature_names"]) == pack["features"].shape[0]


def test_concat_latent_and_proto():
    z = np.array([1.0, 2.0], dtype=np.float32)
    proto = np.array([0.5], dtype=np.float32)
    out = concat_latent_and_proto_features(z, {"features": proto})
    assert out.shape == (3,)


def test_target_prototypes_without_initialized_no_truth_value_error():
    source = np.eye(3, 4, dtype=np.float32)
    target = source + 0.1
    pack = compute_proto_distance_features(
        np.ones(4, dtype=np.float32),
        1,
        source,
        target_prototypes=target,
        mode="own_cancer",
    )
    assert np.all(np.isfinite(pack["features"]))
