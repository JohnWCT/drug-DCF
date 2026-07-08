#!/usr/bin/env python3
"""Round 17R 18-class feature QC unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from tools.extract_round13_proto_features import (
    _assert_18class_mapping,
    _prototype_qc_fields,
)


def test_assert_18class_mapping_accepts_18() -> None:
    mapping = {
        "num_cancer_types": 18,
        "cancer_names": [f"c{i}" for i in range(18)],
        "mapping_source": "checkpoint_metadata",
    }
    _assert_18class_mapping(mapping, require_n=18)


def test_assert_18class_mapping_rejects_28() -> None:
    mapping = {
        "num_cancer_types": 28,
        "cancer_names": [f"c{i}" for i in range(28)],
        "mapping_source": "checkpoint_metadata",
    }
    with pytest.raises(ValueError, match="n_trainable_cancer_types=28"):
        _assert_18class_mapping(mapping, require_n=18)


def test_assert_18class_mapping_rejects_legacy_source() -> None:
    mapping = {
        "num_cancer_types": 18,
        "cancer_names": [f"c{i}" for i in range(18)],
        "mapping_source": "proto_cache",
    }
    with pytest.raises(ValueError, match="checkpoint_metadata"):
        _assert_18class_mapping(mapping, require_n=18)


def test_prototype_qc_fields_flags() -> None:
    mapping = {"num_cancer_types": 18, "cancer_names": [f"c{i}" for i in range(18)]}
    src = np.ones(18, dtype=bool)
    tgt = np.ones(18, dtype=bool)
    qc = _prototype_qc_fields(mapping, src, tgt)
    assert qc["prototype_class_source"] == "checkpoint_metadata"
    assert qc["n_trainable_cancer_types"] == 18
    assert qc["source_prototypes_used"] == 18
    assert qc["target_prototypes_used"] == 18
    assert qc["uses_legacy_28class_cache"] is False
