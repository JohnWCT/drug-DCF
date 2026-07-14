"""Round 18 prediction ensemble helpers (5-fold probability mean)."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


REQUIRED_FOLD_COLS = {
    "eval_row_id",
    "Patient_id",
    "drug_name",
    "Label",
    "fold_id",
    "probability",
    "target_key",
}


def _normalize_fold_predictions(fold_predictions: pd.DataFrame) -> pd.DataFrame:
    df = fold_predictions.copy()
    if "Patient_id" not in df.columns and "ModelID" in df.columns:
        df["Patient_id"] = df["ModelID"].astype(str)
    if "drug_name" not in df.columns and "DRUG_NAME" in df.columns:
        df["drug_name"] = df["DRUG_NAME"].astype(str)
    if "target_key" not in df.columns:
        df["target_key"] = "internal_test"
    if "eval_row_id" not in df.columns:
        raise KeyError(
            "fold_predictions missing eval_row_id; Round 18E requires stable row identity"
        )
    return df


def assert_fold_prediction_uniqueness(
    fold_predictions: pd.DataFrame,
    *,
    required_folds: int = 5,
) -> None:
    """
    Hard uniqueness gates for external ensemble:

    - each eval_row_id × fold_id appears exactly once
    - each eval_row_id has exactly one Label
    - each eval_row_id has required_folds predictions
    - Patient_id / drug_name identity is constant across folds for each eval_row_id
    """
    df = _normalize_fold_predictions(fold_predictions)
    missing = REQUIRED_FOLD_COLS - set(df.columns)
    if missing:
        raise KeyError(f"fold_predictions missing columns: {sorted(missing)}")

    dup = df.duplicated(subset=["eval_row_id", "fold_id"], keep=False)
    if dup.any():
        ex = df.loc[dup, ["eval_row_id", "fold_id"]].head(3)
        raise AssertionError(
            f"eval_row_id × fold_id must be unique; found duplicates e.g.\n{ex}"
        )

    label_n = df.groupby("eval_row_id")["Label"].nunique()
    bad_label = label_n[label_n != 1]
    if len(bad_label):
        raise AssertionError(
            f"eval_row_id must have a single Label; found {len(bad_label)} conflicts"
        )

    fold_n = df.groupby("eval_row_id")["fold_id"].nunique()
    bad_folds = fold_n[fold_n != required_folds]
    if len(bad_folds):
        raise AssertionError(
            f"Expected {required_folds} folds per eval_row_id; "
            f"found {len(bad_folds)} incomplete groups"
        )

    id_n = df.groupby("eval_row_id")["Patient_id"].nunique()
    drug_n = df.groupby("eval_row_id")["drug_name"].nunique()
    if (id_n != 1).any() or (drug_n != 1).any():
        raise AssertionError(
            "Patient_id and drug_name must be identical across folds for each eval_row_id"
        )


def ensemble_fold_predictions(
    fold_predictions: pd.DataFrame,
    *,
    required_folds: int = 5,
    group_cols: Optional[Sequence[str]] = None,
    require_all_folds: bool = True,
) -> pd.DataFrame:
    """
    Aggregate fold-level probabilities by mean.

    Primary group key: target_key + eval_row_id (not Patient×drug×Label alone).
    """
    df = _normalize_fold_predictions(fold_predictions)
    missing = REQUIRED_FOLD_COLS - set(df.columns)
    if missing:
        raise KeyError(f"fold_predictions missing columns: {sorted(missing)}")

    if require_all_folds:
        assert_fold_prediction_uniqueness(df, required_folds=required_folds)

    group_cols = list(group_cols or ["target_key", "eval_row_id"])
    named = {
        "probability": ("probability", "mean"),
        "probability_std": ("probability", "std"),
        "n_folds": ("fold_id", "nunique"),
        "Patient_id": ("Patient_id", "first"),
        "drug_name": ("drug_name", "first"),
        "DRUG_NAME": ("drug_name", "first"),
        "Label": ("Label", "first"),
        "ModelID": ("Patient_id", "first"),
    }
    if "logit" in df.columns:
        named["logit_mean"] = ("logit", "mean")

    grouped = df.groupby(group_cols, sort=False).agg(**named).reset_index()

    if require_all_folds:
        bad = grouped[grouped["n_folds"] != required_folds]
        if len(bad):
            raise AssertionError(
                f"Expected {required_folds} folds for every group; "
                f"found {len(bad)} incomplete groups "
                f"(example n_folds={int(bad['n_folds'].iloc[0])})"
            )
    return grouped


def assert_no_best_fold_selection(fold_predictions: pd.DataFrame) -> None:
    """Guardrail: ensemble must not keep only the best fold."""
    if "selected_fold_only" in fold_predictions.columns:
        if bool(fold_predictions["selected_fold_only"].any()):
            raise AssertionError("best-fold selection is forbidden in Round 18")
