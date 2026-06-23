"""Round 13 prototype-distance response feature QC selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND13_OUTPUT_COLS = (
    "round13_selection_group",
    "round13_selection_reason",
    "round13_proto_feature_score",
    "prototype_feature_mode",
    "response_input_mode",
    "source_model_id",
    "proto_feature_dim",
    "response_input_dim",
)

ROUND13_GROUP_SPECS = (
    ("G1_best_z_only_baseline", "z_only_baseline", False, 2),
    ("G2_best_own_cancer_features", "own_cancer", False, 4),
    ("G3_best_all_source_anchors_features", "all_source_anchors", False, 3),
    ("G4_best_source_target_features", "all_source_and_target", False, 2),
    ("G5_best_own_plus_summary_features", "own_plus_summary", False, 2),
    ("G6_best_round12_exp037_reference", "forced_exp037", True, 1),
    ("G7_best_round11_exp035_reference", "forced_exp035", True, 1),
    ("G8_high_priority_cancer_improvement", "high_priority", False, 2),
    ("G9_fill_ranked", "fill_ranked", False, 99),
)


def _safe_numeric(series, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([series], dtype=float).fillna(default)


def _model_id_col(df: pd.DataFrame) -> str:
    for col in ("Model_ID", "ID", "model_id"):
        if col in df.columns:
            return col
    raise ValueError("No model id column found")


def annotate_round13_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ROUND13_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["Average_TCGA_AUC_mean"] = _safe_numeric(out.get("Average_TCGA_AUC_mean"))
    out["Global_TCGA_AUC_mean"] = _safe_numeric(out.get("Global_TCGA_AUC_mean"))
    out["prototype_feature_mode"] = out.get("prototype_feature_mode", out.get("feature_mode", "none")).astype(str)
    out["response_input_mode"] = out.get(
        "response_input_mode",
        np.where(out["prototype_feature_mode"] == "none", "z_only", "z_plus_proto_features"),
    ).astype(str)
    out["proto_feature_dim"] = _safe_numeric(out.get("proto_feature_dim", 0), default=0.0)
    out["response_input_dim"] = _safe_numeric(out.get("response_input_dim", 0), default=0.0)

    avg = out["Average_TCGA_AUC_mean"]
    glob = out["Global_TCGA_AUC_mean"]
    mode_bonus = np.where(out["prototype_feature_mode"] == "own_cancer", 0.01, 0.0)
    out["round13_proto_feature_score"] = (
        0.75 * avg.fillna(0)
        + 0.20 * glob.fillna(0)
        + mode_bonus
        - np.where(out["prototype_feature_mode"] == "all_source_and_target", 0.005, 0.0)
    )
    return out


def _pick_group(pool: pd.DataFrame, strategy: str, n: int, exclude: set) -> pd.DataFrame:
    id_col = _model_id_col(pool)
    available = pool[~pool[id_col].astype(str).isin(exclude)].copy()
    if available.empty:
        return available.head(0)

    if strategy == "z_only_baseline":
        sub = available[available["prototype_feature_mode"].astype(str) == "none"]
        return sub.sort_values("round13_proto_feature_score", ascending=False, na_position="last").head(n)
    if strategy in ("own_cancer", "all_source_anchors", "all_source_and_target", "own_plus_summary"):
        sub = available[available["prototype_feature_mode"].astype(str) == strategy]
        return sub.sort_values("round13_proto_feature_score", ascending=False, na_position="last").head(n)
    if strategy == "high_priority":
        return available.sort_values("Average_TCGA_AUC_mean", ascending=False, na_position="last").head(n)
    if strategy in ("forced_exp037", "forced_exp035", "fill_ranked"):
        return available.head(0)
    return available.sort_values("round13_proto_feature_score", ascending=False, na_position="last").head(n)


def select_round13_proto_response_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 30,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    force_baseline_models = force_baseline_models or ["r13_exp_037_none", "r13_exp_035_none"]
    id_col = _model_id_col(aggregated_df)
    pool = annotate_round13_scores(aggregated_df)

    selected_ids: set = set()
    selected_rows = []
    group_counts = {}

    forced_patterns = {
        "r13_exp_037_none": ("G6_best_round12_exp037_reference", "forced_exp037"),
        "r13_exp_035_none": ("G7_best_round11_exp035_reference", "forced_exp035"),
    }
    for forced_id in force_baseline_models:
        match = pool[pool[id_col].astype(str) == str(forced_id)]
        if match.empty and "source_model_id" in pool.columns:
            src = "exp_037" if "037" in forced_id else "exp_035"
            match = pool[
                (pool["source_model_id"].astype(str) == src)
                & (pool["prototype_feature_mode"].astype(str) == "none")
            ]
        if not match.empty:
            row = match.iloc[0].copy()
            group, reason = forced_patterns.get(str(forced_id), ("G6_best_round12_exp037_reference", "forced"))
            row["round13_selection_group"] = group
            row["round13_selection_reason"] = reason
            selected_rows.append(row)
            selected_ids.add(str(row[id_col]))

    for group_name, strategy, _forced, quota in ROUND13_GROUP_SPECS:
        if strategy in ("forced_exp037", "forced_exp035"):
            group_counts[group_name] = sum(
                1 for r in selected_rows if r.get("round13_selection_group") == group_name
            )
            continue
        picks = _pick_group(pool, strategy, quota, selected_ids)
        count = 0
        for _, row in picks.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            tagged = row.copy()
            tagged["round13_selection_group"] = group_name
            tagged["round13_selection_reason"] = strategy
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
            "round13_proto_feature_score", ascending=False, na_position="last"
        )
        for _, row in remaining.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            tagged = row.copy()
            tagged["round13_selection_group"] = "G9_fill_ranked"
            tagged["round13_selection_reason"] = "fill_ranked"
            selected_rows.append(tagged)
            selected_ids.add(mid)
            if len(selected_ids) >= top_k:
                break

    if not selected_rows:
        return pool.head(0), {"group_counts": group_counts, "selected": 0}

    result = pd.DataFrame(selected_rows)
    result = annotate_round13_scores(result)
    result["selection_rank"] = range(1, len(result) + 1)
    return result, {"group_counts": group_counts, "selected": len(result)}
