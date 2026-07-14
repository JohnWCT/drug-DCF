"""Round 19 no-hybrid assertions."""
import pytest

from tools.round19_drug_encoders import assert_no_hybrid
from tools.round19_fusion_models import assert_compatible


def test_no_hybrid_maccs_graph():
    with pytest.raises(AssertionError):
        assert_no_hybrid("maccs", has_maccs=True, has_graph=True)
    with pytest.raises(AssertionError):
        assert_no_hybrid("gin", has_maccs=True, has_graph=True)
    assert_no_hybrid("maccs", has_maccs=True, has_graph=False)
    assert_no_hybrid("gine", has_maccs=False, has_graph=True)


def test_compatibility_rejects_d4_p2_and_d1_p2():
    with pytest.raises(AssertionError):
        assert_compatible("D4", "P2")
    with pytest.raises(AssertionError):
        assert_compatible("D1", "P2")
    assert_compatible("D2", "P2")
    assert_compatible("D4", "P1")
