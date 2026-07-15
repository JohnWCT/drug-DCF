import ast
from pathlib import Path

from tools import round19_stage19f_role_selector as selector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND_ROOT = PROJECT_ROOT / "result" / "optimization_runs" / "round19_factorial"


def _proposal():
    return selector.build_proposal(ROUND_ROOT, project_root=PROJECT_ROOT)


def test_selector_proposes_expected_roles_without_single_champion():
    proposal = _proposal()

    assert proposal["proposal_only"] is True
    assert proposal["single_champion"] is None
    roles = proposal["roles"]
    assert proposal["lock_type"] == "round19_final_role_proposal"
    assert roles["historical_anchor"]["candidate_id"] == "E0"
    assert roles["source_performance_champion"]["candidate_id"] == "F2"
    assert roles["parsimonious_context_model"]["candidate_id"] == "F1"
    assert roles["cancer_shift_specialist"]["candidate_id"] == "E1"
    assert roles["chemical_shift_specialist"]["candidate_id"] == "E3"
    assert roles["source_only_domain_candidate"]["candidate_id"] == "E4"
    assert roles["efficient_model"]["candidate_id"] == "E5"
    assert roles["general_recommended_model"]["candidate_id"] == "E3"
    assert proposal["selection_used_internal"] is False
    assert proposal["selection_used_tcga"] is False
    assert proposal["completion_evidence"]["stage19d"]["completed_jobs"] == 90
    assert proposal["completion_evidence"]["stage19e"]["completed_jobs"] == 90
    assert proposal["completion_evidence"]["stage19e"]["failed_jobs"] == 0
    assert proposal["final_role_lock_created"] is False


def test_general_guardrail_requires_all_shifts_and_no_major_failure():
    proposal = _proposal()
    evidence = proposal["raw_precision_shift_evidence"]

    assert all(
        evidence["E3"][shift]["status"] in selector.PASSING for shift in selector.SHIFTS
    )
    assert not any(evidence["E3"][shift]["major_fail"] for shift in selector.SHIFTS)
    assert evidence["E1"]["drug_heldout"]["major_fail"] is True
    assert proposal["role_evidence"]["general"]["eligible_candidates"] == ["E0", "E3"]
    assert selector.classify_shift(0.003)["status"] == "PASS"
    assert selector.classify_shift(-0.003)["status"] == "FAIL"
    assert selector.classify_shift(-0.015)["major_fail"] is True


def test_chemical_specialist_uses_maximin_before_mean_delta():
    def row(delta, *, major=False):
        return {
            "delta_vs_E0": delta,
            "major_fail": major,
            "mean_DrugMacro_AUPRC": 0.4,
            "std_DrugMacro_AUC": 0.01,
        }

    shifts = {
        "E0": {
            "drug_heldout": row(0.004),
            "scaffold_heldout": row(-0.002),
        },
        "E3": {
            "drug_heldout": row(0.001),
            "scaffold_heldout": row(0.0),
        },
        "E4": {
            "drug_heldout": row(0.5, major=True),
            "scaffold_heldout": row(0.5),
        },
    }
    policy = {"simplicity_order": {"E0": 0, "E3": 1, "E4": 2}}

    assert selector.select_chemical_specialist(shifts, policy) == "E3"


def test_no_external_selection_input_paths_are_executable():
    source_path = PROJECT_ROOT / "tools" / "round19_stage19f_role_selector.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = ("internal", "tcga", "integrated5", "external")
    path_calls = {"open", "_read_csv", "_read_json", "_sha256"}

    executable_paths = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name not in path_calls:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                executable_paths.append(arg.value.lower())

    assert not any(term in path for path in executable_paths for term in forbidden)
    assert set(selector.REPORT_FILES) == {
        "stage19d_cross_summary",
        "stage19d_paired",
        "stage19d_resource",
        "stage19e_per_shift",
        "stage19e_guardrails",
        "stage19e_paired",
        "stage19e_resource",
        "stage19d_experiment_lock",
        "stage19e_candidate_lock",
        "stage19e_experiment_lock",
    }
