"""Round 14 VICReg latent stabilizer QC selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.round7_selection import is_vicreg_active

ROUND14_OUTPUT_COLS = (
    "round14_selection_group",
    "round14_selection_reason",
    "round14_vicreg_stabilizer_score",
    "round14_route_id",
    "round14_vicreg_active",
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
)

ROUND14_GROUP_SPECS = (
    ("G1_best_pretrain_composite", "composite", False, 3),
    ("G2_best_exp008_route", "exp008_route", False, 3),
    ("G3_best_exp035_route", "exp035_route", False, 3),
    ("G4_best_low_vicreg", "low_vicreg", False, 2),
    ("G5_best_mid_vicreg", "mid_vicreg", False, 2),
    ("G6_best_no_vicreg_control", "no_vicreg", False, 2),
    ("G7_best_active_dims", "active_dims", False, 2),
    ("G8_best_covariance_stability", "cov_stability", False, 2),
    ("G9_best_proto_gap_safe", "proto_gap_safe", False, 2),
    ("G10_fill_ranked", "fill_ranked", False, 99),
)

COLLAPSE_KMEANS_FLOOR = 0.35
COLLAPSE_ACTIVE_DIMS_FLOOR = 4


def _safe_numeric(series, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([series], dtype=float).fillna(default)


def _model_id_col(df: pd.DataFrame) -> str:
    for col in ("ID", "model_id", "exp_id"):
        if col in df.columns:
            return col
    raise ValueError("No model id column found")


def _route_id(row: pd.Series) -> str:
    for col in ("route_id", "round14_route_id"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val)
    branch = str(row.get("round14_branch", ""))
    source = str(row.get("source_model", row.get("source_baseline_exp_id", "")))
    if "008" in source or branch == "14B":
        return "exp008_proto_response_route"
    if "035" in source or branch == "14C":
        return "exp035_strong_zonly_route"
    return ""


def _vicreg_lambda_sum(row: pd.Series) -> float:
    lv = pd.to_numeric(row.get("lambda_tumor_var", row.get("final_gan_g_lambda_tumor_var_eff", 0)), errors="coerce")
    lc = pd.to_numeric(row.get("lambda_tumor_cov", row.get("final_gan_g_lambda_tumor_cov_eff", 0)), errors="coerce")
    return float((lv if pd.notna(lv) else 0.0) + (lc if pd.notna(lc) else 0.0))


def _collapse_risk(row: pd.Series) -> bool:
    ari = pd.to_numeric(row.get("kmeans_ari"), errors="coerce")
    active = pd.to_numeric(row.get("latent_active_dims"), errors="coerce")
    if pd.notna(ari) and float(ari) < COLLAPSE_KMEANS_FLOOR:
        return True
    # Only apply active-dim floor when the metric was recorded (not missing → NaN).
    if pd.notna(active) and float(active) > 0 and float(active) < COLLAPSE_ACTIVE_DIMS_FLOOR:
        return True
    return False


def annotate_round14_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ROUND14_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["round14_vicreg_active"] = out.apply(is_vicreg_active, axis=1)
    out["round14_route_id"] = out.apply(_route_id, axis=1)
    out["lambda_tumor_var"] = _safe_numeric(out.get("lambda_tumor_var", 0), default=0.0)
    out["lambda_tumor_cov"] = _safe_numeric(out.get("lambda_tumor_cov", 0), default=0.0)
    out["latent_active_dims"] = _safe_numeric(
        out.get("latent_active_dims", out.get("active_dims", np.nan)), default=np.nan
    )
    out["latent_cov_offdiag_mean"] = _safe_numeric(
        out.get("latent_cov_offdiag_mean", out.get("tumor_vicreg_cov_offdiag_mean_abs", np.nan))
    )
    out["kmeans_ari"] = _safe_numeric(out.get("kmeans_ari"))
    out["wasserstein"] = _safe_numeric(out.get("wasserstein"))
    out["mean_target_to_source_anchor_distance"] = _safe_numeric(
        out.get("mean_target_to_source_anchor_distance", out.get("mean_same_cancer_proto_distance"))
    )
    out["mean_conditional_leakage_strength"] = _safe_numeric(
        out.get("mean_conditional_leakage_strength", out.get("conditional_leakage_strength"))
    )

    sweet = _safe_numeric(out.get("sweetspot_tcga_proxy_score", out.get("score_total", 0)), default=0.0)
    active_bonus = np.clip(out["latent_active_dims"] / 32.0, 0.0, 1.0) * 0.10
    cov_penalty = np.clip(out["latent_cov_offdiag_mean"].fillna(0.5), 0.0, 1.0) * 0.05
    if "source_anchor_proto_enabled" in out.columns:
        proto_enabled = out["source_anchor_proto_enabled"].astype(str).str.lower().isin(["true", "1"])
    else:
        proto_enabled = pd.Series(False, index=out.index)
    proto_bonus = np.where(
        proto_enabled,
        np.clip(1.0 - out["mean_target_to_source_anchor_distance"].fillna(1.0), 0.0, 1.0) * 0.05,
        0.0,
    )
    collapse_penalty = np.where(out.apply(_collapse_risk, axis=1), 0.25, 0.0)

    out["round14_vicreg_stabilizer_score"] = (
        0.55 * sweet
        + 0.15 * out["kmeans_ari"].fillna(0.0)
        + active_bonus
        + proto_bonus
        - cov_penalty
        - collapse_penalty
        + np.where(out["round14_vicreg_active"], 0.02, 0.0)
    )
    return out


def _pick_group(pool: pd.DataFrame, strategy: str, n: int, exclude: set) -> pd.DataFrame:
    id_col = _model_id_col(pool)
    available = pool[~pool[id_col].astype(str).isin(exclude)].copy()
    available = available[~available.apply(_collapse_risk, axis=1)]
    if available.empty:
        return available.head(0)

    if strategy == "composite":
        return available.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    if strategy == "exp008_route":
        sub = available[available["round14_route_id"].astype(str) == "exp008_proto_response_route"]
        return sub.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    if strategy == "exp035_route":
        sub = available[available["round14_route_id"].astype(str) == "exp035_strong_zonly_route"]
        return sub.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    if strategy == "low_vicreg":
        sub = available[available["round14_vicreg_active"].fillna(False)]
        sub = sub.assign(_lam=available.apply(_vicreg_lambda_sum, axis=1))
        sub = sub[(sub["_lam"] > 0) & (sub["_lam"] <= 0.00006)]
        return sub.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    if strategy == "mid_vicreg":
        sub = available[available["round14_vicreg_active"].fillna(False)]
        sub = sub.assign(_lam=available.apply(_vicreg_lambda_sum, axis=1))
        sub = sub[(sub["_lam"] > 0.00006) & (sub["_lam"] <= 0.0004)]
        return sub.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    if strategy == "no_vicreg":
        sub = available[~available["round14_vicreg_active"].fillna(False)]
        return sub.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    if strategy == "active_dims":
        return available.sort_values("latent_active_dims", ascending=False, na_position="last").head(n)
    if strategy == "cov_stability":
        return available.sort_values("latent_cov_offdiag_mean", ascending=True, na_position="last").head(n)
    if strategy == "proto_gap_safe":
        sub = available.copy()
        sub["_gap"] = sub["mean_target_to_source_anchor_distance"].fillna(1.0)
        return sub.sort_values("_gap", ascending=True, na_position="last").head(n)
    if strategy == "fill_ranked":
        return available.sort_values("round14_vicreg_stabilizer_score", ascending=False, na_position="last").head(n)
    return available.head(0)


def select_round14_vicreg_stabilizer_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 16,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    force_baseline_models = force_baseline_models or []
    id_col = _model_id_col(aggregated_df)
    pool = annotate_round14_scores(aggregated_df)

    selected_ids: set = set()
    selected_rows = []
    group_counts = {}

    for group_name, strategy, _forced, quota in ROUND14_GROUP_SPECS:
        if len(selected_rows) >= top_k:
            break
        picks = _pick_group(pool, strategy, quota, selected_ids)
        for _, row in picks.iterrows():
            mid = str(row[id_col])
            if mid in selected_ids:
                continue
            row = row.copy()
            row["round14_selection_group"] = group_name
            row["round14_selection_reason"] = strategy
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
            row["round14_selection_group"] = "G10_fill_ranked"
            row["round14_selection_reason"] = "fill_ranked"
            selected_rows.append(row)
            selected_ids.add(mid)
            if len(selected_rows) >= top_k:
                break

    top_df = pd.DataFrame(selected_rows)
    if not top_df.empty:
        top_df["selection_rank"] = range(1, len(top_df) + 1)

    info = {
        "top_k": top_k,
        "selected_count": len(top_df),
        "group_counts": group_counts,
        "forced_baselines_requested": force_baseline_models,
    }
    return top_df, info
