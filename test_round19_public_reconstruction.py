"""Public-reconstruction schema/registry/release adapter tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.round19_deployment_policy_export import (
    Round19DeploymentRouter,
    build_deployment_policy,
)
from tools.round19_registry import (
    COMPATIBLE_CELLS,
    PUBLIC_MANUAL_COMPATIBLE_CELLS,
    validate_registry_invariants,
)
from tools.round19_role_lock import load_role_lock, summarize_role_lock
from tools.round19_schema import (
    Round19JobSpec,
    Round19ModelSpec,
    canonical_job_id,
    canonical_model_id,
    validate_manifest_columns,
    validate_model_spec,
    validate_selection_input_columns,
)

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"


def test_registry_has_exactly_thirteen_executed_cells():
    assert len(COMPATIBLE_CELLS) == 13
    assert len(set(COMPATIBLE_CELLS)) == 13
    result = validate_registry_invariants()
    assert result["ok"] is True
    # Local executed matrix differs from public-manual ideal.
    assert ("D4", "P1") in COMPATIBLE_CELLS
    assert ("D4", "P2") not in COMPATIBLE_CELLS
    assert ("D4", "P0") in PUBLIC_MANUAL_COMPATIBLE_CELLS


def test_schema_rejects_incompatible_and_forbidden_columns():
    with pytest.raises(AssertionError):
        validate_model_spec(
            Round19ModelSpec(omics_id="O1", drug_id="D4", predictor_id="P2")
        )
    with pytest.raises(AssertionError):
        validate_selection_input_columns(["DrugMacro_AUC", "TCGA_Integrated5"])
    with pytest.raises(AssertionError):
        validate_selection_input_columns(["posthoc_score"])
    validate_selection_input_columns(["DrugMacro_AUC", "Global_AUPRC"])
    validate_manifest_columns(
        ["job_id", "drug_representation_id", "omics_id", "predictor_id", "fold_id"]
    )


def test_canonical_ids_are_deterministic():
    model = Round19ModelSpec(omics_id="O2", drug_id="D0", predictor_id="P2")
    assert canonical_model_id(model) == "O2__D0__P2__pure"
    job = Round19JobSpec(
        stage="19b",
        model=model,
        fold_id=0,
        model_seed=101,
        split_seed=42,
    )
    assert "19b__D0__P2__O2__split42__seed101__fold0" in canonical_job_id(job)


def test_role_lock_and_deployment_policy_agree():
    assert LOCK.is_file(), f"missing lock at {LOCK}"
    lock = load_role_lock(LOCK)
    summary = summarize_role_lock(lock)
    assert summary["immutable"] is True
    assert summary["checkpoint_inventory_count"] == 90
    assert summary["selection_used_tcga"] is False
    policy = build_deployment_policy(role_lock=lock)
    router = Round19DeploymentRouter(lock, policy)
    cancer = router.route({}, novelty_class="unseen_cancer_type")
    chemical = router.route({}, novelty_class="unseen_drug")
    source = router.route({}, novelty_class="source_like")
    assert cancer.role == "cancer_shift_specialist"
    assert chemical.role == "chemical_shift_specialist"
    assert source.role == "source_performance_champion"
    assert cancer.candidate_id is not None
    assert chemical.candidate_id is not None
