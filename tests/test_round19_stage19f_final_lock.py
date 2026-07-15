
from pathlib import Path

import pytest

from tools.round19_stage19f_final_lock import (
    build_final_lock,
    verify_final_lock,
    write_exclusive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "result/optimization_runs/round19_factorial/reports"


def build():
    return build_final_lock(
        proposal_path=REPORTS / "round19_final_role_proposal.json",
        inventory_path=REPORTS / "round19_stage19f_checkpoint_inventory.csv",
        inventory_summary_path=REPORTS / "round19_stage19f_checkpoint_inventory_summary.json",
        policy_path=PROJECT_ROOT / "config/round19_stage19f_role_policy.json",
        inference_settings_path=PROJECT_ROOT / "config/round19_stage19f_inference_settings.json",
        approved_by="human-reviewer",
        approved_at_utc="2026-07-15T14:44:00Z",
        review_reference="round19f-proposal-review",
        notes="approved in test",
        project_root=PROJECT_ROOT,
    )


def test_final_lock_hashes_all_members_and_verifies():
    lock = build()
    assert lock["immutable"] is True
    assert lock["approval_metadata"]["decision"] == "approved"
    assert len(lock["hashes"]["checkpoint_inventory"]) == 90
    assert lock["role_immutability"]["internal_test_may_change_roles"] is False
    assert lock["role_immutability"]["tcga_may_change_roles"] is False
    verify_final_lock(lock, PROJECT_ROOT)


def test_final_lock_is_exclusive_create(tmp_path):
    lock = build()
    output = tmp_path / "lock.json"
    write_exclusive(lock, output)
    with pytest.raises(FileExistsError):
        write_exclusive(lock, output)


def test_tampered_lock_is_rejected():
    lock = build()
    lock["roles"]["chemical_shift_specialist"]["candidate_id"] = "E0"
    with pytest.raises(AssertionError, match="payload hash"):
        verify_final_lock(lock, PROJECT_ROOT)
