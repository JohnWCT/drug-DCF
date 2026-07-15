"""Strict 15-member atom-attention ensemble helpers for Stage 19G."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tools.round19_stage19f_ensemble import N_REQUIRED_MEMBERS, REQUIRED_MEMBER_IDS


IDENTITY_COLUMNS = (
    "candidate_id",
    "eval_row_id",
    "member_id",
    "atom_index",
)


def validate_attention_long(
    attention: pd.DataFrame,
    *,
    required_members: Sequence[str] = REQUIRED_MEMBER_IDS,
) -> pd.DataFrame:
    df = attention.copy()
    if "attention_kind" in df:
        df = df[df["attention_kind"].astype(str) == "primary"].copy()
        if df.empty:
            raise AssertionError("attention export contains no primary rows")
    required = {
        *IDENTITY_COLUMNS,
        "attention",
        "is_valid_atom",
        "graph_smiles",
        "ModelID",
        "drug_name",
        "target_key",
    }
    missing = required - set(df)
    if missing:
        raise KeyError(f"attention rows missing columns: {sorted(missing)}")
    if df.duplicated(list(IDENTITY_COLUMNS)).any():
        raise AssertionError("candidate/eval/member/atom attention rows must be unique")
    values = pd.to_numeric(df["attention"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError("attention must be finite and non-negative")
    df["attention"] = values.astype(float)
    expected = set(map(str, required_members))
    if len(expected) != N_REQUIRED_MEMBERS:
        raise AssertionError("Stage 19G requires the fixed 15-member grid")
    for key, group in df.groupby(["candidate_id", "eval_row_id"], sort=False):
        members = set(group["member_id"].astype(str))
        if members != expected:
            raise AssertionError(
                f"{key} requires all 15 members; missing={sorted(expected-members)} "
                f"extra={sorted(members-expected)}"
            )
        atom_sets = group.groupby("member_id")["atom_index"].agg(
            lambda x: tuple(sorted(map(int, x)))
        )
        if atom_sets.nunique() != 1:
            raise AssertionError(f"Atom coverage differs across members for {key}")
        for column in ("graph_smiles", "ModelID", "drug_name", "target_key"):
            if group[column].nunique(dropna=False) != 1:
                raise AssertionError(f"{column} differs across members for {key}")
    return df


def ensemble_atom_attention(attention: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic mean of primary attention from exactly 15 members."""
    df = validate_attention_long(attention)
    keys = ["candidate_id", "eval_row_id", "atom_index"]
    result = (
        df.groupby(keys, sort=False, dropna=False)
        .agg(
            attention=("attention", "mean"),
            attention_std=("attention", lambda x: float(np.std(x, ddof=0))),
            n_members=("member_id", "nunique"),
            is_valid_atom=("is_valid_atom", "first"),
            graph_smiles=("graph_smiles", "first"),
            ModelID=("ModelID", "first"),
            drug_name=("drug_name", "first"),
            target_key=("target_key", "first"),
        )
        .reset_index()
    )
    if not (result["n_members"] == N_REQUIRED_MEMBERS).all():
        raise AssertionError("Attention ensemble is not complete")
    return result


__all__ = ["ensemble_atom_attention", "validate_attention_long"]
