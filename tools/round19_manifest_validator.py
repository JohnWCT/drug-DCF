"""Validate Round 19 manifests against compatibility + population identity."""
from __future__ import annotations

from typing import Iterable, Sequence, Set

import pandas as pd

from tools.round19_fusion_models import COMPATIBLE_CELLS, assert_compatible


FORBIDDEN_SELECTION_COLS = {
    "TCGA_AUC",
    "Integrated5",
    "external_auc",
    "internal_test_auc",
    "Integrated5_DrugMacro_TCGA_AUC",
}


def validate_compatible_manifest(df: pd.DataFrame) -> None:
    required = {
        "drug_representation_id",
        "predictor_id",
        "omics_id",
        "fold_id",
        "job_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    for _, row in df.iterrows():
        assert_compatible(str(row["drug_representation_id"]), str(row["predictor_id"]))
        if str(row.get("encoder_type", "")) == "maccs" and bool(row.get("has_graph", False)):
            raise AssertionError("Hybrid MACCS+graph row in manifest")


def assert_expected_job_count(df: pd.DataFrame, expected: int, *, label: str = "manifest") -> None:
    n = int(len(df))
    if n != int(expected):
        raise AssertionError(f"{label} expected {expected} jobs, got {n}")


def assert_same_population(drug_sets: Sequence[Set[str]], *, label: str = "drugs") -> None:
    if not drug_sets:
        return
    base = drug_sets[0]
    for i, s in enumerate(drug_sets[1:], start=1):
        if s != base:
            raise AssertionError(f"{label} set mismatch at index {i}: {len(s)} vs {len(base)}")


def assert_selection_frame_has_no_tcga(df: pd.DataFrame) -> None:
    hits = sorted(FORBIDDEN_SELECTION_COLS.intersection(df.columns))
    if hits:
        raise AssertionError(f"Selection frame contains forbidden external columns: {hits}")
