"""Round 19 data / manifest / scaffold / lock smoke tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.round19_fusion_models import COMPATIBLE_CELLS
from tools.round19_manifest_validator import (
    assert_expected_job_count,
    assert_selection_frame_has_no_tcga,
    validate_compatible_manifest,
)
from tools.round19_scaffold_groups import murcko_scaffold_id
from tools.round19_selection_lock import scan_mapping_for_forbidden, write_selection_lock


def test_scaffold_ids_stable():
    a = murcko_scaffold_id("CCO")
    b = murcko_scaffold_id("CCO")
    assert a == b
    assert isinstance(a, str) and len(a) > 0


def test_selection_lock_rejects_tcga_keys(tmp_path):
    with pytest.raises(AssertionError):
        scan_mapping_for_forbidden({"metrics": {"TCGA_AUC": 0.1}})
    path = write_selection_lock(
        {"selection_uses_tcga": False, "candidates": [{"id": "x", "mean_DrugMacro_AUC": 0.5}]},
        str(tmp_path / "lock.json"),
    )
    assert path.is_file()


def test_selection_frame_no_tcga_columns():
    df = pd.DataFrame({"architecture_id": ["a"], "mean_DrugMacro_AUC": [0.5]})
    assert_selection_frame_has_no_tcga(df)
    with pytest.raises(AssertionError):
        assert_selection_frame_has_no_tcga(pd.DataFrame({"Integrated5": [1.0]}))


def test_stage19b_manifest_count_if_present():
    path = Path("result/optimization_runs/round19_factorial/manifests/stage19b_drug_predictor_manifest.csv")
    if not path.is_file():
        pytest.skip("19b manifest not built yet")
    df = pd.read_csv(path)
    validate_compatible_manifest(df)
    assert_expected_job_count(df, len(COMPATIBLE_CELLS) * 2 * 3)
