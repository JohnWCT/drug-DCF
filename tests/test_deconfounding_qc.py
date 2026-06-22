import pytest
from tools.round9_diagnostics_common import classify_deconfounding_qc, leakage_strength

def test_leakage_strength_symmetric():
    assert leakage_strength(0.3) == pytest.approx(0.2)
    assert leakage_strength(0.7) == pytest.approx(0.2)

def test_global_only_alignment_status():
    status = classify_deconfounding_qc(0.52, 0.8, 0.5, 0.3)
    assert status == "global_only_alignment"

def test_biology_collapse_risk_status():
    status = classify_deconfounding_qc(0.52, 0.52, 0.1, 0.01)
    assert status == "biology_collapse_risk"

def test_good_conditional_deconfounding_status():
    status = classify_deconfounding_qc(0.52, 0.52, 0.5, 0.3)
    assert status == "good_conditional_deconfounding"
