"""19E experiment lock forbidden-field tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.round19_selection_lock import scan_mapping_for_forbidden

ROOT = Path("result/optimization_runs/round19_factorial")


@pytest.mark.integration
def test_stage19e_experiment_lock_clean():
    path = ROOT / "reports" / "round19_stage19e_experiment_lock.json"
    if not path.is_file():
        pytest.skip("experiment lock missing")
    payload = json.loads(path.read_text())
    assert payload["lock_type"] == "stage19e_experiment_lock"
    assert payload["internal_test_used"] is False
    assert payload["tcga_used"] is False
    scan_mapping_for_forbidden(payload)
