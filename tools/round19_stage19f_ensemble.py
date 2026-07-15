"""Strict 15-member probability ensemble for Round 19F."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_SEEDS = (52, 62, 72)
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_MEMBER_IDS = tuple(
    f"seed{seed}_fold{fold}" for seed in REQUIRED_SEEDS for fold in REQUIRED_FOLDS
)
N_REQUIRED_MEMBERS = len(REQUIRED_MEMBER_IDS)


def member_id(seed: int, fold: int) -> str:
    seed_i, fold_i = int(seed), int(fold)
    if seed_i not in REQUIRED_SEEDS or fold_i not in REQUIRED_FOLDS:
        raise ValueError(f"Unsupported Round 19F member: seed={seed_i}, fold={fold_i}")
    return f"seed{seed_i}_fold{fold_i}"


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    if "candidate_id" not in df and "source_candidate_id" in df:
        df["candidate_id"] = df["source_candidate_id"]
    if "ModelID" not in df and "Patient_id" in df:
        df["ModelID"] = df["Patient_id"]
    if "drug_name" not in df and "DRUG_NAME" in df:
        df["drug_name"] = df["DRUG_NAME"]
    if "target_key" not in df and "target" in df:
        df["target_key"] = df["target"]
    if "member_id" not in df and {"split_seed", "fold_id"} <= set(df.columns):
        df["member_id"] = [
            member_id(seed, fold)
            for seed, fold in zip(df["split_seed"], df["fold_id"])
        ]
    return df


def validate_ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate completeness, uniqueness, identity, and the fixed member grid."""
    df = _normalize_predictions(predictions)
    required = {
        "candidate_id",
        "eval_row_id",
        "member_id",
        "Label",
        "ModelID",
        "drug_name",
        "target_key",
        "probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"predictions missing columns: {sorted(missing)}")
    if "selected_fold_only" in df and df["selected_fold_only"].fillna(False).astype(bool).any():
        raise AssertionError("best-fold selection is forbidden in Round 19F")
    if df[list(required)].isna().any().any():
        bad = [c for c in required if df[c].isna().any()]
        raise AssertionError(f"prediction metadata contains nulls: {sorted(bad)}")

    pair_cols = ["candidate_id", "eval_row_id", "member_id"]
    dup = df.duplicated(pair_cols, keep=False)
    if dup.any():
        example = df.loc[dup, pair_cols].iloc[0].to_dict()
        raise AssertionError(f"candidate × eval_row_id × member_id must be unique: {example}")

    group_cols = ["candidate_id", "eval_row_id"]
    expected = set(REQUIRED_MEMBER_IDS)
    for key, group in df.groupby(group_cols, sort=False, dropna=False):
        members = set(group["member_id"].astype(str))
        if len(group) != N_REQUIRED_MEMBERS or members != expected:
            missing_members = sorted(expected - members)
            extra_members = sorted(members - expected)
            raise AssertionError(
                f"{key} requires exactly {N_REQUIRED_MEMBERS} members; "
                f"got {len(group)}, missing={missing_members}, extra={extra_members}"
            )
        for column in ("Label", "ModelID", "drug_name", "target_key"):
            if group[column].nunique(dropna=False) != 1:
                raise AssertionError(
                    f"{column} inconsistent across members for candidate/eval row {key}"
                )
    for eval_row_id, group in df.groupby("eval_row_id", sort=False, dropna=False):
        for column in ("Label", "ModelID", "drug_name", "target_key"):
            if group[column].nunique(dropna=False) != 1:
                raise AssertionError(
                    f"{column} inconsistent across candidates for eval_row_id={eval_row_id}"
                )

    probability = pd.to_numeric(df["probability"], errors="coerce")
    if probability.isna().any() or not np.isfinite(probability.to_numpy()).all():
        raise AssertionError("probability must be finite numeric values")
    if ((probability < 0) | (probability > 1)).any():
        raise AssertionError("probability must be in [0, 1]")
    df["probability"] = probability.astype(float)
    return df


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean all 15 member probabilities; no fold/member selection is allowed."""
    df = validate_ensemble_predictions(predictions)
    group_cols = ["candidate_id", "eval_row_id"]
    aggregated = (
        df.groupby(group_cols, sort=False, dropna=False)
        .agg(
            probability=("probability", "mean"),
            probability_std=("probability", lambda values: float(np.std(values, ddof=0))),
            n_members=("member_id", "nunique"),
            Label=("Label", "first"),
            ModelID=("ModelID", "first"),
            drug_name=("drug_name", "first"),
            target_key=("target_key", "first"),
        )
        .reset_index()
    )
    if not (aggregated["n_members"] == N_REQUIRED_MEMBERS).all():
        raise AssertionError("Round 19F output contains a non-15-member ensemble")
    aggregated["DRUG_NAME"] = aggregated["drug_name"]
    return aggregated


class Round19Stage19FEnsemble:
    required_seeds: Sequence[int] = REQUIRED_SEEDS
    required_folds: Sequence[int] = REQUIRED_FOLDS

    @staticmethod
    def validate(predictions: pd.DataFrame) -> pd.DataFrame:
        return validate_ensemble_predictions(predictions)

    @staticmethod
    def aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
        return ensemble_predictions(predictions)


# Explicit alias for callers migrating from the Round 18 helper name.
ensemble_member_predictions = ensemble_predictions
