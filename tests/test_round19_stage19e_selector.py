"""Stage 19E candidate lock tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.round19_stage19e_selector import build_candidate_lock

ROOT = Path("result/optimization_runs/round19_factorial")


@pytest.mark.integration
def test_stage19e_selector_mandatory_and_pins():
    if not (ROOT / "reports" / "round19_stage19d_experiment_lock.json").is_file():
        pytest.skip("19D experiment lock missing")
    lock = build_candidate_lock(ROOT)
    ids = {c["candidate_id"] for c in lock["candidates"]}
    assert {"E0", "E1", "E2", "E3", "E4"}.issubset(ids)
    by = {c["candidate_id"]: c for c in lock["candidates"]}
    assert by["E1"]["source_candidate_id"] == "F1_primary_o2"
    assert by["E2"]["source_candidate_id"] == "F2_full_omics_o3"
    assert by["E4"]["source_candidate_id"] == "F4_source_only_o4"
    assert by["E1"]["omics_id"] == "O2"
    assert by["E2"]["omics_id"] == "O3"
    assert by["E4"]["omics_id"] == "O4"
    assert lock["internal_test_used"] is False
    assert lock["tcga_used"] is False
    assert "E5" in ids
