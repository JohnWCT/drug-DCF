"""Round 18 prediction ensemble helpers (5-fold probability mean)."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


REQUIRED_FOLD_COLS = {
    "Patient_id",
    "drug_name",
    "Label",
    "fold_id",
    "probability",
    "target_key",
}


def ensemble_fold_predictions(
    fold_predictions: pd.DataFrame,
    *,
    required_folds: int = 5,
    group_cols: Optional[Sequence[str]] = None,
    require_all_folds: bool = True,
) -> pd.DataFrame:
    """
    Aggregate fold-level probabilities by mean.

    Asserts every group has exactly required_folds unique fold_ids when
    require_all_folds=True (Round 18 TCGA / internal-test rule).
    """
    df = fold_predictions.copy()
    if "Patient_id" not in df.columns and "ModelID" in df.columns:
        df["Patient_id"] = df["ModelID"].astype(str)
    if "drug_name" not in df.columns and "DRUG_NAME" in df.columns:
        df["drug_name"] = df["DRUG_NAME"].astype(str)
    if "target_key" not in df.columns:
        df["target_key"] = "internal_test"

    missing = REQUIRED_FOLD_COLS - set(df.columns)
    if missing:
        raise KeyError(f"fold_predictions missing columns: {sorted(missing)}")

    group_cols = list(group_cols or ["target_key", "Patient_id", "drug_name", "Label"])
    named = {
        "probability": ("probability", "mean"),
        "probability_std": ("probability", "std"),
        "n_folds": ("fold_id", "nunique"),
    }
    if "logit" in df.columns:
        named["logit_mean"] = ("logit", "mean")

    grouped = df.groupby(group_cols, sort=False).agg(**named).reset_index()

    if require_all_folds:
        bad = grouped[grouped["n_folds"] != required_folds]
        if len(bad):
            raise AssertionError(
                f"Expected {required_folds} folds for every group; "
                f"found {len(bad)} incomplete groups (example n_folds={int(bad['n_folds'].iloc[0])})"
            )
    return grouped


def assert_no_best_fold_selection(fold_predictions: pd.DataFrame) -> None:
    """Guardrail: ensemble must not keep only the best fold."""
    if "selected_fold_only" in fold_predictions.columns:
        if bool(fold_predictions["selected_fold_only"].any()):
            raise AssertionError("best-fold selection is forbidden in Round 18")
