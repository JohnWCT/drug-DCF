from tools.round20.completion_audit import run_completion_audit
from tools.round20.lock_consistency import recalculate_stage20a_decision, recalculate_stage20b_guardrails


def test_stage20a_decision_recalculates(synthetic_run_root):
    recalc = recalculate_stage20a_decision(run_root=synthetic_run_root)
    assert recalc["selected_context"] == "C32"
    assert recalc["stored_decision_matches_recalculation"]


def test_stage20b_guardrails_recalculate(synthetic_run_root):
    recalc = recalculate_stage20b_guardrails(run_root=synthetic_run_root)
    assert recalc["all_pass"] is False
    assert recalc["stored_guardrails_match_recalculation"]


def test_completion_audit_passes_on_synthetic(synthetic_run_root):
    audit = run_completion_audit(run_root=synthetic_run_root, strict=False, write_artifacts=False)
    assert audit["audit_status"] == "PASS"
