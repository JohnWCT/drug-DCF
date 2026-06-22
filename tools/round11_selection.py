"""Round 11 stability + reconstruction QC selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND11_OUTPUT_COLS = (
    "round11_selection_group",
    "round11_selection_reason",
    "round11_stability_score",
    "round11_branch",
    "reconstruction_loss_type",
    "smooth_l1_beta",
    "lambda_cond_adv",
    "global_adv_mode",
    "macro_conditional_domain_auc",
    "mean_conditional_leakage_strength",
    "kmeans_ari",
    "wasserstein",
    "fid",
    "biology_collapse_risk",
)

ROUND11_GROUP_SPECS = (
    ("G1_best_downstream_proxy", "downstream_proxy", False, 4),
    ("G2_low_conditional_leakage", "low_leakage", False, 4),
    ("G3_best_10c_stabilized", "best_10c", False, 4),
    ("G4_best_smooth_l1", "best_smooth_l1", False, 4),
    ("G5_best_hybrid_reconstruction", "best_hybrid", False, 3),
    ("G6_best_mse_control", "mse_control", False, 3),
    ("G7_global_alignment_safe", "global_alignment", False, 3),
    ("G8_cancer_retention_safe", "cancer_retention", False, 3),
    ("G9_forced_exp111", "forced_reference", True, 2),
    ("G10_fill_ranked", "fill_ranked", False, 99),
)


def _safe_numeric(series, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([series], dtype=float).fillna(default)


def _column_or_default(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return _safe_numeric(df[col], default=default)
    return pd.Series(default, index=df.index, dtype=float)


def _normalize_higher_better(values: pd.Series) -> pd.Series:
    vals = _safe_numeric(values)
    if vals.notna().sum() == 0:
        return pd.Series(0.0, index=vals.index)
    vmin, vmax = vals.min(), vals.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax <= vmin:
        return pd.Series(0.5, index=vals.index)
    return (vals - vmin) / (vmax - vmin)


def _normalize_lower_better(values: pd.Series) -> pd.Series:
    return 1.0 - _normalize_higher_better(values)


def annotate_round11_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ROUND11_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["mean_conditional_leakage_strength"] = _safe_numeric(
        out.get("mean_conditional_leakage_strength", out.get("conditional_leakage_strength"))
    )
    out["macro_conditional_domain_auc"] = _safe_numeric(
        out.get("macro_conditional_domain_auc", out.get("conditional_domain_auc_macro"))
    )
    out["kmeans_ari"] = _safe_numeric(out.get("kmeans_ari"))
    out["wasserstein"] = _safe_numeric(out.get("wasserstein"))
    out["fid"] = _safe_numeric(out.get("fid"))
    out["round11_branch"] = out.get("round11_branch", "").astype(str)
    out["reconstruction_loss_type"] = out.get("reconstruction_loss_type", "mse").astype(str)

    out["biology_collapse_risk"] = (out["kmeans_ari"] < 0.30).fillna(False)

    leakage_score = _normalize_lower_better(out["mean_conditional_leakage_strength"])
    retention_score = _normalize_higher_better(out["kmeans_ari"])
    alignment_score = (
        0.5 * _normalize_lower_better(out["wasserstein"])
        + 0.5 * _normalize_lower_better(out["fid"])
    )
    recon_bonus = np.where(
        out["reconstruction_loss_type"].str.contains("smooth_l1|hybrid", regex=True, na=False),
        0.05,
        0.0,
    )

    out["round11_stability_score"] = (
        0.25 * leakage_score
        + 0.25 * retention_score
        + 0.20 * alignment_score
        + 0.15 * _normalize_higher_better(_column_or_default(out, "round10_cond_adv_score", 0))
        + 0.15 * _normalize_higher_better(_column_or_default(out, "Average_TCGA_AUC_mean", 0))
        + recon_bonus
    )
    out.loc[out["biology_collapse_risk"], "round11_stability_score"] *= 0.25
    return out


def _model_id_col(df: pd.DataFrame) -> str:
    for col in ("ID", "model_id", "exp_id", "experiment_id"):
        if col in df.columns:
            return col
    raise ValueError("No model id column found")


def _pick_group(pool: pd.DataFrame, strategy: str, n: int, exclude: set) -> pd.DataFrame:
    id_col = _model_id_col(pool)
    available = pool[~pool[id_col].astype(str).isin(exclude)].copy()
    if available.empty:
        return available

    if strategy == "downstream_proxy":
        sort_col = "Average_TCGA_AUC_mean" if "Average_TCGA_AUC_mean" in available.columns else "round11_stability_score"
        return available.sort_values(sort_col, ascending=False, na_position="last").head(n)
    if strategy == "low_leakage":
        return available.sort_values("mean_conditional_leakage_strength", ascending=True, na_position="last").head(n)
    if strategy == "best_10c":
        sub = available[available["round11_branch"].str.contains("11B_10C|11C_10C", regex=True, na=False)]
        return sub.sort_values("round11_stability_score", ascending=False, na_position="last").head(n)
    if strategy == "best_smooth_l1":
        sub = available[available["reconstruction_loss_type"] == "smooth_l1"]
        return sub.sort_values("round11_stability_score", ascending=False, na_position="last").head(n)
    if strategy == "best_hybrid":
        sub = available[available["reconstruction_loss_type"] == "hybrid_mse_smooth_l1"]
        return sub.sort_values("round11_stability_score", ascending=False, na_position="last").head(n)
    if strategy == "mse_control":
        sub = available[available["reconstruction_loss_type"] == "mse"]
        return sub.sort_values("round11_stability_score", ascending=False, na_position="last").head(n)
    if strategy == "global_alignment":
        return available.sort_values("wasserstein", ascending=True, na_position="last").head(n)
    if strategy == "cancer_retention":
        return available.sort_values("kmeans_ari", ascending=False, na_position="last").head(n)
    if strategy == "forced_reference":
        return available.head(0)
    return available.sort_values("round11_stability_score", ascending=False, na_position="last").head(n)


def select_round11_stability_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 30,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    force_baseline_models = force_baseline_models or ["exp_111"]
    id_col = _model_id_col(aggregated_df)
    pool = annotate_round11_scores(aggregated_df)

    selected_ids: set = set()
    selected_rows = []
    group_counts = {}

    for model_id in force_baseline_models:
        match = pool[pool[id_col].astype(str) == str(model_id)]
        if not match.empty:
            row = match.iloc[0].copy()
            row["round11_selection_group"] = "G9_forced_exp111"
            row["round11_selection_reason"] = "forced_reference"
            selected_rows.append(row)
            selected_ids.add(str(model_id))

    for group_name, strategy, _forced, quota in ROUND11_GROUP_SPECS:
        if group_name == "G9_forced_exp111":
            group_counts[group_name] = sum(
                1 for r in selected_rows if r.get("round11_selection_group") == group_name
            )
            continue
        picks = _pick_group(pool, strategy, quota, selected_ids)
        count = 0
        for _, row in picks.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            tagged = row.copy()
            tagged["round11_selection_group"] = group_name
            tagged["round11_selection_reason"] = strategy
            selected_rows.append(tagged)
            selected_ids.add(mid)
            count += 1
            if len(selected_ids) >= top_k:
                break
        group_counts[group_name] = count
        if len(selected_ids) >= top_k:
            break

    if len(selected_ids) < top_k:
        remaining = pool[~pool[id_col].astype(str).isin(selected_ids)].sort_values(
            "round11_stability_score", ascending=False, na_position="last"
        )
        for _, row in remaining.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            tagged = row.copy()
            tagged["round11_selection_group"] = "G10_fill_ranked"
            tagged["round11_selection_reason"] = "fill_ranked"
            selected_rows.append(tagged)
            selected_ids.add(mid)
            group_counts["G10_fill_ranked"] = group_counts.get("G10_fill_ranked", 0) + 1
            if len(selected_ids) >= top_k:
                break

    top_df = pd.DataFrame(selected_rows).head(top_k)
    info = {
        "group_counts": group_counts,
        "total_selected": len(top_df),
        "ranking_primary_metric": "round11_stability_score",
        "selection_mode": "round11_stability_qc",
        "forced_baseline_models": force_baseline_models,
    }
    return top_df, info
