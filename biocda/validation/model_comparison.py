"""Paired model comparison utilities."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def paired_model_deltas(
    summary_df: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in metric_columns:
        for model_a, model_b in pairs:
            a = summary_df[summary_df["model"] == model_a].set_index("seed")
            b = summary_df[summary_df["model"] == model_b].set_index("seed")
            shared = sorted(set(a.index) & set(b.index))
            for seed in shared:
                rows.append(
                    {
                        "split_seed": int(seed),
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "delta": float(a.loc[seed, metric] - b.loc[seed, metric]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_paired_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return delta_df
    return (
        delta_df.groupby(["metric", "model_a", "model_b"])["delta"]
        .agg(["mean", "median", "min", lambda s: int((s > 0).sum())])
        .rename(columns={"<lambda_0>": "positive_seed_count"})
        .reset_index()
    )
