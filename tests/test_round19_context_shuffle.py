"""Tests for Round 19C context shuffle controls."""

from __future__ import annotations

import numpy as np
import pytest

from tools.round19_context_controls import (
    BASE_SEED,
    apply_context_permutation,
    build_modelid_derangement,
    build_partition_permutation,
    context_slice_for_omics,
    shuffle_seeds_for_fold,
    validate_context_shuffle,
)


def test_shuffle_seeds_for_fold():
    tr, va = shuffle_seeds_for_fold(0)
    assert tr == BASE_SEED + 1
    assert va == BASE_SEED + 2
    tr2, va2 = shuffle_seeds_for_fold(1)
    assert tr2 == BASE_SEED + 101
    assert va2 == BASE_SEED + 102


def test_derangement_no_self_mapping():
    ids = [f"M{i}" for i in range(8)]
    perm = build_modelid_derangement(ids, seed=42)
    validate_context_shuffle(perm, ids)
    assert set(perm.keys()) == set(ids)
    assert set(perm.values()) == set(ids)


def test_derangement_single_raises():
    with pytest.raises(ValueError):
        build_modelid_derangement(["only"], seed=1)


def test_partition_permutation_matches_derangement():
    ids = ["A", "B", "C", "D"]
    p1 = build_partition_permutation(ids, 7)
    p2 = build_modelid_derangement(ids, 7)
    assert p1 == p2


def test_apply_context_o2_keeps_z_swaps_context():
    z_end, ctx_end = context_slice_for_omics("O2")
    assert z_end == 64 and ctx_end == 80
    vec = np.arange(80, dtype=np.float32)
    donor = vec.copy()
    donor[64:80] = np.linspace(100, 115, 16, dtype=np.float32)
    out = apply_context_permutation(vec, donor, "O2")
    assert np.allclose(out[:64], vec[:64])
    assert np.allclose(out[64:80], donor[64:80])
    assert not np.allclose(out[64:80], vec[64:80])


def test_apply_context_o3_preserves_summary():
    vec = np.arange(91, dtype=np.float32)
    donor = vec.copy()
    donor[64:80] = 42.0
    out = apply_context_permutation(vec, donor, "O3")
    assert np.allclose(out[:64], vec[:64])
    assert np.allclose(out[64:80], 42.0)
    assert np.allclose(out[80:], vec[80:])
