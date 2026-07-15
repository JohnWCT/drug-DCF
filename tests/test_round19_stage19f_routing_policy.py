
import pytest

from tools.round19_deployment_policy import route, select_role


def test_deterministic_scenario_routes():
    assert select_role("unseen_drug") == "chemical_shift_specialist"
    assert select_role("unseen_scaffold") == "chemical_shift_specialist"
    assert select_role("unseen_cancer_type") == "cancer_shift_specialist"
    assert select_role("source_like") == "source_performance_champion"
    decision = route({"novelty_class": "metadata_unknown", "confidence": "low"})
    assert decision.selected_role == "chemical_shift_specialist"
    assert decision.conservative_fallback is True
    assert decision.confidence == "low"


def test_unknown_class_fails_instead_of_confidence_gating():
    with pytest.raises(ValueError):
        select_role("high_model_confidence")
