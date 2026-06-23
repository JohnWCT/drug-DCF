"""Round 12 source-anchor prototype alignment QC selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND12_OUTPUT_COLS = (
    "round12_selection_group",
    "round12_selection_reason",
    "round12_proto_alignment_score",
    "round12_branch",
    "source_anchor_proto_enabled",
    "lambda_proto_align",
    "proto_align_metric",
    "mean_same_cancer_proto_distance",
    "inter_cancer_margin",
    "macro_conditional_domain_auc",
    "mean_conditional_leakage_strength",
    "kmeans_ari",
    "wasserstein",
    "fid",
    "reconstruction_loss_type",
    "biology_collapse_risk",
)

ROUND12_GROUP_SPECS = (
    ("G1_best_proto_gap_reduction", "proto_gap_reduction", False, 4),
    ("G2_best_conditional_leakage_safe", "low_leakage", False, 4),
    ("G3_best_inter_cancer_margin_safe", "inter_margin", False, 3),
    ("G4_best_10C_proto_main", "main_12b", False, 4),
    ("G5_best_low_lambda_proto", "low_lambda", False, 3),
    ("G6_best_mid_lambda_proto", "mid_lambda", False, 3),
    ("G7_best_hybrid_or_smoothl1_proto", "hybrid_smoothl1", False, 3),
    ("G8_best_mse_control", "mse_control", False, 3),
    ("G9_forced_exp035", "forced_exp035", True, 1),
    ("G10_forced_exp111", "forced_exp111", True, 1),
    ("G11_fill_ranked", "fill_ranked", False, 99),
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


def _proto_distance_col(df: pd.DataFrame) -> pd.Series:
    for col in (
        "mean_same_cancer_proto_distance",
        "mean_same_cancer_source_target_cosine_distance",
        "weighted_same_cancer_proto_distance",
    ):
        if col in df.columns:
            return _safe_numeric(df[col])
    return pd.Series(np.nan, index=df.index, dtype=float)


def annotate_round12_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ROUND12_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["mean_same_cancer_proto_distance"] = _proto_distance_col(out)
    out["inter_cancer_margin"] = _safe_numeric(
        out.get(
            "inter_cancer_margin",
            out.get("mean_inter_cancer_target_margin", out.get("mean_inter_cancer_source_margin")),
        )
    )
    out["mean_conditional_leakage_strength"] = _safe_numeric(
        out.get("mean_conditional_leakage_strength", out.get("conditional_leakage_strength"))
    )
    out["macro_conditional_domain_auc"] = _safe_numeric(
        out.get("macro_conditional_domain_auc", out.get("conditional_domain_auc_macro"))
    )
    out["kmeans_ari"] = _safe_numeric(out.get("kmeans_ari"))
    out["wasserstein"] = _safe_numeric(out.get("wasserstein"))
    out["fid"] = _safe_numeric(out.get("fid"))
    out["lambda_proto_align"] = _safe_numeric(out.get("lambda_proto_align", 0.0), default=0.0)
    out["round12_branch"] = out.get("round12_branch", "").astype(str)
    out["reconstruction_loss_type"] = out.get("reconstruction_loss_type", "mse").astype(str)
    if "source_anchor_proto_enabled" in out.columns:
        out["source_anchor_proto_enabled"] = out["source_anchor_proto_enabled"].fillna(False)
    else:
        out["source_anchor_proto_enabled"] = out["lambda_proto_align"].fillna(0.0) > 0

    out["biology_collapse_risk"] = (out["kmeans_ari"] < 0.30).fillna(False)

    proto_gap_score = _normalize_lower_better(out["mean_same_cancer_proto_distance"])
    leakage_score = _normalize_lower_better(out["mean_conditional_leakage_strength"])
    margin_score = _normalize_higher_better(out["inter_cancer_margin"])
    retention_score = _normalize_higher_better(out["kmeans_ari"])
    alignment_score = (
        0.5 * _normalize_lower_better(out["wasserstein"])
        + 0.5 * _normalize_lower_better(out["fid"])
    )

    out["round12_proto_alignment_score"] = (
        0.30 * proto_gap_score
        + 0.20 * leakage_score
        + 0.20 * margin_score
        + 0.15 * retention_score
        + 0.10 * alignment_score
        + 0.05 * _normalize_higher_better(_column_or_default(out, "Average_TCGA_AUC_mean", 0))
    )
    out.loc[out["biology_collapse_risk"], "round12_proto_alignment_score"] *= 0.25
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

    if strategy == "proto_gap_reduction":
        return available.sort_values(
            "mean_same_cancer_proto_distance", ascending=True, na_position="last"
        ).head(n)
    if strategy == "low_leakage":
        return available.sort_values(
            "mean_conditional_leakage_strength", ascending=True, na_position="last"
        ).head(n)
    if strategy == "inter_margin":
        return available.sort_values("inter_cancer_margin", ascending=False, na_position="last").head(n)
    if strategy == "main_12b":
        sub = available[available["round12_branch"].str.contains("12B_proto", na=False)]
        return sub.sort_values("round12_proto_alignment_score", ascending=False, na_position="last").head(n)
    if strategy == "low_lambda":
        sub = available[
            (available["lambda_proto_align"] > 0)
            & (available["lambda_proto_align"] <= 0.0003)
        ]
        return sub.sort_values("round12_proto_alignment_score", ascending=False, na_position="last").head(n)
    if strategy == "mid_lambda":
        sub = available[
            (available["lambda_proto_align"] >= 0.001)
            & (available["lambda_proto_align"] <= 0.003)
        ]
        return sub.sort_values("round12_proto_alignment_score", ascending=False, na_position="last").head(n)
    if strategy == "hybrid_smoothl1":
        sub = available[
            available["reconstruction_loss_type"].str.contains("smooth_l1|hybrid", regex=True, na=False)
        ]
        return sub.sort_values("round12_proto_alignment_score", ascending=False, na_position="last").head(n)
    if strategy == "mse_control":
        sub = available[
            (available["reconstruction_loss_type"] == "mse")
            & (~available["source_anchor_proto_enabled"].fillna(False))
        ]
        return sub.sort_values("round12_proto_alignment_score", ascending=False, na_position="last").head(n)
    if strategy in ("forced_exp035", "forced_exp111", "fill_ranked"):
        return available.head(0)
    return available.sort_values("round12_proto_alignment_score", ascending=False, na_position="last").head(n)


def select_round12_proto_alignment_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 30,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    force_baseline_models = force_baseline_models or ["exp_035", "exp_111"]
    id_col = _model_id_col(aggregated_df)
    pool = annotate_round12_scores(aggregated_df)

    selected_ids: set = set()
    selected_rows = []
    group_counts = {}

    forced_map = {
        "exp_035": ("G9_forced_exp035", "forced_exp035"),
        "exp_111": ("G10_forced_exp111", "forced_exp111"),
    }
    for model_id in force_baseline_models:
        match = pool[pool[id_col].astype(str) == str(model_id)]
        if not match.empty:
            row = match.iloc[0].copy()
            group, reason = forced_map.get(str(model_id), ("G9_forced_exp035", "forced_reference"))
            row["round12_selection_group"] = group
            row["round12_selection_reason"] = reason
            selected_rows.append(row)
            selected_ids.add(str(model_id))

    for group_name, strategy, _forced, quota in ROUND12_GROUP_SPECS:
        if strategy in ("forced_exp035", "forced_exp111"):
            group_counts[group_name] = sum(
                1 for r in selected_rows if r.get("round12_selection_group") == group_name
            )
            continue
        picks = _pick_group(pool, strategy, quota, selected_ids)
        count = 0
        for _, row in picks.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            tagged = row.copy()
            tagged["round12_selection_group"] = group_name
            tagged["round12_selection_reason"] = strategy
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
            "round12_proto_alignment_score", ascending=False, na_position="last"
        )
        for _, row in remaining.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            tagged = row.copy()
            tagged["round12_selection_group"] = "G11_fill_ranked"
            tagged["round12_selection_reason"] = "fill_ranked"
            selected_rows.append(tagged)
            selected_ids.add(mid)
            if len(selected_ids) >= top_k:
                break

    if not selected_rows:
        return pool.head(0), {"group_counts": group_counts, "selected": 0}

    result = pd.DataFrame(selected_rows)
    result = annotate_round12_scores(result)
    result["selection_rank"] = range(1, len(result) + 1)
    return result, {"group_counts": group_counts, "selected": len(result)}
