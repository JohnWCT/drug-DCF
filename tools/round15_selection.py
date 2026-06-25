"""Round 15 reproducibility + exp_008 route rescue QC selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.round14_selection import (
    annotate_round14_scores,
    _collapse_risk,
    _model_id_col,
    _route_id,
    _vicreg_lambda_sum,
)
from tools.round7_selection import is_vicreg_active

ROUND15_OUTPUT_COLS = (
    "round15_selection_group",
    "round15_selection_reason",
    "round15_repro_rescue_score",
    "round15_route_id",
    "round15_branch",
    "round15_vicreg_active",
    "lambda_tumor_var",
    "lambda_tumor_cov",
    "tumor_vicreg_start_epoch",
    "tumor_vicreg_full_epoch",
    "latent_active_dims",
    "latent_cov_offdiag_mean",
    "kmeans_ari",
    "wasserstein",
    "mean_target_to_source_anchor_distance",
    "mean_conditional_leakage_strength",
    "source_model",
)

ROUND15_GROUP_SPECS = (
    ("G1_round13_best_repro", "exp008_route", False, 2),
    ("G2_forced_exp008_route", "exp008_route", True, 2),
    ("G3_best_exp008_low_vicreg", "low_vicreg", False, 2),
    ("G4_best_exp008_no_vicreg", "no_vicreg", True, 2),
    ("G5_round14_best_exp078", "exp035_route", False, 1),
    ("G6_round13_exp035_reference", "exp035_route", False, 1),
    ("G7_best_own_plus_summary_candidate", "composite", False, 1),
    ("G8_fill_ranked", "fill_ranked", False, 99),
)


def _safe_numeric(series, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([series], dtype=float).fillna(default)


def _round15_branch(row: pd.Series) -> str:
    for col in ("round15_branch", "round14_branch"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val)
    return "15C"


def _round15_route_id(row: pd.Series) -> str:
    for col in ("route_id", "round15_route_id", "round14_route_id"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val)
    branch = str(row.get("round15_branch", row.get("round14_branch", "")))
    source = str(row.get("source_model", row.get("source_baseline_exp_id", "")))
    if branch in ("15C", "14B") or "008" in source:
        return "exp008_proto_response_route"
    if branch in ("14C",) or "035" in source:
        return "exp035_strong_zonly_route"
    return _route_id(row)


def annotate_round15_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = annotate_round14_scores(df)
    for col in ROUND15_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["round15_branch"] = out.apply(_round15_branch, axis=1)
    out["round15_route_id"] = out.apply(_round15_route_id, axis=1)
    out["round15_vicreg_active"] = out.apply(is_vicreg_active, axis=1)
    out["source_model"] = out.get("source_model", out.get("source_baseline_exp_id", ""))

    sweet = _safe_numeric(out.get("sweetspot_tcga_proxy_score", out.get("score_total", 0)), default=0.0)
    collapse_penalty = np.where(out.apply(_collapse_risk, axis=1), 0.20, 0.0)
    exp008_bonus = np.where(
        out["round15_route_id"].astype(str) == "exp008_proto_response_route",
        0.05,
        0.0,
    )
    no_vicreg_bonus = np.where(~out["round15_vicreg_active"].fillna(False), 0.02, 0.0)

    out["round15_repro_rescue_score"] = (
        0.50 * sweet
        + 0.20 * out["kmeans_ari"].fillna(0.0)
        + exp008_bonus
        + no_vicreg_bonus
        - collapse_penalty
        + 0.10 * out.get("round14_vicreg_stabilizer_score", 0).fillna(0.0)
    )
    return out


def _pick_group(pool: pd.DataFrame, strategy: str, n: int, exclude: set, forced: bool = False) -> pd.DataFrame:
    id_col = _model_id_col(pool)
    available = pool[~pool[id_col].astype(str).isin(exclude)].copy()
    if not forced:
        available = available[~available.apply(_collapse_risk, axis=1)]
    if available.empty and forced:
        available = pool[~pool[id_col].astype(str).isin(exclude)].copy()
    if available.empty:
        return available.head(0)

    score_col = "round15_repro_rescue_score"

    if strategy == "composite":
        return available.sort_values(score_col, ascending=False, na_position="last").head(n)
    if strategy == "exp008_route":
        sub = available[available["round15_route_id"].astype(str) == "exp008_proto_response_route"]
        if sub.empty:
            sub = available[
                available["source_model"].astype(str).str.contains("008", na=False)
                | available["round15_branch"].astype(str).str.contains("15C|14B", regex=True, na=False)
            ]
        return sub.sort_values(score_col, ascending=False, na_position="last").head(n)
    if strategy == "exp035_route":
        sub = available[available["round15_route_id"].astype(str) == "exp035_strong_zonly_route"]
        return sub.sort_values(score_col, ascending=False, na_position="last").head(n)
    if strategy == "low_vicreg":
        sub = available[available["round15_vicreg_active"].fillna(False)].copy()
        sub = sub.assign(_lam=available.apply(_vicreg_lambda_sum, axis=1))
        sub = sub[(sub["_lam"] > 0) & (sub["_lam"] <= 0.00006)]
        sub = sub[sub["round15_route_id"].astype(str) == "exp008_proto_response_route"]
        return sub.sort_values(score_col, ascending=False, na_position="last").head(n)
    if strategy == "no_vicreg":
        sub = available[~available["round15_vicreg_active"].fillna(False)]
        sub = sub[sub["round15_route_id"].astype(str) == "exp008_proto_response_route"]
        return sub.sort_values(score_col, ascending=False, na_position="last").head(n)
    if strategy == "fill_ranked":
        return available.sort_values(score_col, ascending=False, na_position="last").head(n)
    return available.head(0)


def _force_exp008_controls(pool: pd.DataFrame, selected_ids: set, selected_rows: List[pd.Series]) -> None:
    id_col = _model_id_col(pool)
    exp008 = pool[pool["round15_route_id"].astype(str) == "exp008_proto_response_route"]
    if exp008.empty:
        exp008 = pool[pool["source_model"].astype(str).str.contains("008", na=False)]
    if exp008.empty:
        return

    no_vicreg = exp008[~exp008["round15_vicreg_active"].fillna(False)]
    pick_pool = no_vicreg if not no_vicreg.empty else exp008
    pick = pick_pool.sort_values("round15_repro_rescue_score", ascending=False, na_position="last")
    for _, row in pick.iterrows():
        mid = str(row[id_col])
        if mid in selected_ids:
            continue
        row = row.copy()
        row["round15_selection_group"] = "G4_best_exp008_no_vicreg_forced"
        row["round15_selection_reason"] = "force_exp008_route_control"
        selected_rows.append(row)
        selected_ids.add(mid)
        break


def select_round15_repro_rescue_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 12,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    force_baseline_models = force_baseline_models or []
    id_col = _model_id_col(aggregated_df)
    pool = annotate_round15_scores(aggregated_df)

    selected_ids: set = set()
    selected_rows: List[pd.Series] = []
    group_counts = {}

    _force_exp008_controls(pool, selected_ids, selected_rows)

    for group_name, strategy, forced, quota in ROUND15_GROUP_SPECS:
        if len(selected_rows) >= top_k:
            break
        picks = _pick_group(pool, strategy, quota, selected_ids, forced=forced)
        for _, row in picks.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            row = row.copy()
            row["round15_selection_group"] = group_name
            row["round15_selection_reason"] = strategy
            selected_rows.append(row)
            selected_ids.add(mid)
            group_counts[group_name] = group_counts.get(group_name, 0) + 1
            if len(selected_rows) >= top_k:
                break

    if len(selected_rows) < top_k:
        fill = _pick_group(pool, "fill_ranked", top_k - len(selected_rows), selected_ids)
        for _, row in fill.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            row = row.copy()
            row["round15_selection_group"] = "G8_fill_ranked"
            row["round15_selection_reason"] = "fill_ranked"
            selected_rows.append(row)
            selected_ids.add(mid)
            if len(selected_rows) >= top_k:
                break

    if not any(
        str(r.get("round15_route_id", r.get("round14_route_id", ""))) == "exp008_proto_response_route"
        for r in selected_rows
    ):
        _force_exp008_controls(pool, selected_ids, selected_rows)

    top_df = pd.DataFrame(selected_rows)
    if not top_df.empty:
        top_df["selection_rank"] = range(1, len(top_df) + 1)
        if "route_id" not in top_df.columns:
            top_df["route_id"] = top_df["round15_route_id"]

    info = {
        "top_k": top_k,
        "selected_count": len(top_df),
        "group_counts": group_counts,
        "forced_baselines_requested": force_baseline_models,
        "exp008_route_included": bool(
            not top_df.empty
            and (top_df["round15_route_id"].astype(str) == "exp008_proto_response_route").any()
        ),
    }
    return top_df, info
