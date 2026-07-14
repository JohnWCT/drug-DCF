"""Tests for Round 19C candidate selector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.round19_stage19c_selector import (
    ROLE_CANDIDATES,
    ROLE_FIXED,
    deduplicate_cells,
    select_role_cells,
)


def _synthetic_scores() -> pd.DataFrame:
    rows = []
    grid = {
        ("D0", "P0"): (0.62, 0.61, 0.40, 0.39),
        ("D0", "P1"): (0.60, 0.61, 0.38, 0.39),
        ("D0", "P2"): (0.63, 0.62, 0.41, 0.40),
        ("D1", "P0"): (0.55, 0.54, 0.30, 0.29),
        ("D2", "P0"): (0.58, 0.59, 0.33, 0.34),
        ("D3", "P0"): (0.57, 0.56, 0.32, 0.31),
        ("D4", "P0"): (0.56, 0.55, 0.31, 0.30),
        ("D1", "P1"): (0.57, 0.56, 0.32, 0.31),
        ("D2", "P1"): (0.59, 0.60, 0.34, 0.35),
        ("D3", "P1"): (0.58, 0.57, 0.33, 0.32),
        ("D4", "P1"): (0.61, 0.60, 0.36, 0.35),
        ("D2", "P2"): (0.60, 0.61, 0.35, 0.36),
        ("D3", "P2"): (0.62, 0.63, 0.37, 0.38),
    }
    for (d, p), (o2, o3, a2, a3) in grid.items():
        rows.append(
            {
                "drug_id": d,
                "predictor_id": p,
                "mean_auc_o2": o2,
                "mean_auc_o3": o3,
                "mean_auc_o2_o3": (o2 + o3) / 2,
                "mean_auprc_o2_o3": (a2 + a3) / 2,
            }
        )
    return pd.DataFrame(rows)


def test_fixed_roles():
    scores = _synthetic_scores()
    cells = select_role_cells(scores)
    by_role = {c["role"]: c for c in cells}
    for role, (d, p) in ROLE_FIXED.items():
        assert by_role[role]["drug_id"] == d
        assert by_role[role]["predictor_id"] == p


def test_role_candidate_pools():
    scores = _synthetic_scores()
    cells = select_role_cells(scores)
    by_role = {c["role"]: c for c in cells}
    assert (by_role["R3"]["drug_id"], by_role["R3"]["predictor_id"]) in ROLE_CANDIDATES["R3"]
    assert (by_role["R4"]["drug_id"], by_role["R4"]["predictor_id"]) in ROLE_CANDIDATES["R4"]
    assert (by_role["R5"]["drug_id"], by_role["R5"]["predictor_id"]) in ROLE_CANDIDATES["R5"]
    assert (by_role["R6"]["drug_id"], by_role["R6"]["predictor_id"]) in ROLE_CANDIDATES["R6"]


def test_deduplicate_cells():
    scores = _synthetic_scores()
    cells = select_role_cells(scores)
    unique, mapping = deduplicate_cells(cells)
    keys = {(u["drug_id"], u["predictor_id"]) for u in unique}
    assert len(keys) == len(unique)
    assert len(mapping) == len(cells)


@pytest.mark.integration
def test_selector_on_real_root():
    root = Path("result/optimization_runs/round19_factorial")
    manifest = root / "manifests" / "stage19b_drug_predictor_manifest.csv"
    if not manifest.is_file():
        pytest.skip("real round19 factorial root not present")
    from tools.round19_stage19c_selector import build_candidate_lock

    lock = build_candidate_lock(root, require_complete=True, expected_jobs=117)
    assert lock["lock_type"] == "stage19c_candidate_lock"
    assert lock["stage19b_completed_jobs"] == 117
    roles = {c["role"] for c in lock["selected_cells"]}
    assert {"R0", "R1", "R2"}.issubset(roles)
    assert lock["internal_test_used"] is False
