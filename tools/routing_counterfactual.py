#!/usr/bin/env python3
"""Post-lock descriptive routing counterfactuals and regret (never role selection)."""
from __future__ import annotations

import pandas as pd


CLASSIFICATION = "post_lock_descriptive"


def routing_regret(
    predictions: pd.DataFrame,
    *,
    case_column: str = "case_id",
    role_column: str = "role",
    loss_column: str = "loss",
    selected_column: str = "selected_role",
) -> pd.DataFrame:
    required = {case_column, role_column, loss_column, selected_column}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"counterfactual input missing: {sorted(missing)}")
    rows = []
    for case_id, group in predictions.groupby(case_column, sort=False):
        selected = group[selected_column].iloc[0]
        selected_rows = group[group[role_column] == selected]
        if len(selected_rows) != 1:
            raise AssertionError("each case requires exactly one locked selected-role result")
        selected_loss = float(selected_rows[loss_column].iloc[0])
        oracle_loss = float(group[loss_column].min())
        rows.append({
            case_column: case_id,
            "selected_role": selected,
            "selected_loss": selected_loss,
            "descriptive_oracle_loss": oracle_loss,
            "descriptive_regret": selected_loss - oracle_loss,
            "classification": CLASSIFICATION,
            "may_change_roles": False,
        })
    return pd.DataFrame(rows)
