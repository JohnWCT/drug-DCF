"""Round 19 omics composition tests (O2/O3/O4)."""
import pickle

import numpy as np
import pytest

from tools.round19_feature_builder import (
    OMICS_ALIAS,
    build_o2_from_o3,
    build_o4_source_only_stub,
    default_r17_feature_root,
)


@pytest.fixture(scope="module")
def feature_root():
    root = default_r17_feature_root()
    if not (root / "own_proto_context_projected_16" / "ccle_latent_proto.pkl").is_file():
        pytest.skip("Round 17R feature artifacts not available")
    return root


def test_o2_excludes_summary_and_matches_o3_prefix(feature_root, tmp_path):
    meta = build_o2_from_o3(feature_root, tmp_path)
    assert meta["response_input_dim"] == 80
    assert meta["includes_own_plus_summary"] is False
    with open(tmp_path / OMICS_ALIAS["O2"] / "ccle_latent_proto.pkl", "rb") as f:
        o2 = pickle.load(f)
    with open(feature_root / "own_proto_context_projected_16" / "ccle_latent_proto.pkl", "rb") as f:
        o3 = pickle.load(f)
    mid = next(iter(o2))
    assert o2[mid].shape == (80,)
    assert np.allclose(o2[mid], o3[mid][:80])


def test_o4_has_no_target_fields(feature_root, tmp_path):
    meta = build_o4_source_only_stub(feature_root, tmp_path)
    assert meta["includes_target_prototype_fields"] is False
    names = (tmp_path / OMICS_ALIAS["O4"] / "feature_names.json").read_text().lower()
    assert "target" not in names
