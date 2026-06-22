"""Round 10 conditional adversarial deconfounding selection."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND10_OUTPUT_COLS = (
    "round10_selection_group",
    "round10_selection_reason",
    "round10_cond_adv_score",
    "round10_branch",
    "lambda_cond_adv",
    "cancer_condition_dim",
    "global_adv_mode",
    "macro_conditional_domain_auc",
    "mean_conditional_leakage_strength",
    "cancer_classifier_macro_f1",
    "inter_cancer_margin",
    "collapse_flag",
    "biology_collapse_risk",
)

ROUND10_GROUP_SPECS = (
    ("G1_lowest_conditional_leakage", "lowest_leakage", False, 4),
    ("G2_best_leakage_improvement_vs_10A", "leakage_improvement", False, 4),
    ("G3_good_cancer_retention", "cancer_retention", False, 3),
    ("G4_safe_global_alignment", "global_alignment", False, 3),
    ("G5_best_10B_replacement", "best_10B", False, 3),
    ("G6_best_10C_weak_global_guard", "best_10C", False, 2),
    ("G7_lambda_diversity", "lambda_diversity", False, 4),
    ("G8_condition_dim_diversity", "dim_diversity", False, 3),
    ("G9_forced_baseline", "forced_baseline", True, 2),
    ("G10_fill_ranked", "fill_ranked", False, 99),
)

LAMBDA_FORCE_VALUES = [0.0001, 0.0003, 0.001, 0.003]


def _safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


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


def annotate_round10_scores(df: pd.DataFrame, baseline_10a: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df.copy()
    for col in ROUND10_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["macro_conditional_domain_auc"] = _safe_numeric(
        out.get("macro_conditional_domain_auc", out.get("conditional_domain_auc_macro"))
    )
    out["mean_conditional_leakage_strength"] = _safe_numeric(
        out.get("mean_conditional_leakage_strength", out.get("conditional_leakage_strength"))
    )
    out["kmeans_ari"] = _safe_numeric(out.get("kmeans_ari"))
    out["wasserstein"] = _safe_numeric(out.get("wasserstein"))
    out["fid"] = _safe_numeric(out.get("fid"))
    out["inter_cancer_margin"] = _safe_numeric(out.get("inter_cancer_margin"))
    out["cancer_classifier_macro_f1"] = _safe_numeric(
        out.get("cancer_classifier_macro_f1", out.get("cancer_macro_f1"))
    )
    out["collapse_flag"] = out.get("collapse_flag", False).fillna(False).astype(bool)

    baseline_leakage = np.nan
    if baseline_10a is not None and not baseline_10a.empty:
        baseline_leakage = _safe_numeric(
            baseline_10a.get("mean_conditional_leakage_strength")
        ).mean()
    if pd.isna(baseline_leakage):
        mask_10a = out.get("round10_branch", "").astype(str).str.contains("10A", na=False)
        if mask_10a.any():
            baseline_leakage = out.loc[mask_10a, "mean_conditional_leakage_strength"].mean()

    leakage_improve = baseline_leakage - out["mean_conditional_leakage_strength"]
    out["conditional_leakage_improvement"] = leakage_improve

    out["biology_collapse_risk"] = (
        (out["kmeans_ari"] < 0.30)
        | (out["inter_cancer_margin"] < out["inter_cancer_margin"].quantile(0.10))
        | out["collapse_flag"]
    ).fillna(False)

    out["conditional_leakage_improvement_score"] = _normalize_higher_better(leakage_improve)
    out["cancer_retention_score"] = _normalize_higher_better(out["kmeans_ari"])
    out["prototype_margin_score"] = _normalize_higher_better(out["inter_cancer_margin"])
    out["global_alignment_safety_score"] = (
        0.5 * _normalize_lower_better(out["wasserstein"])
        + 0.5 * _normalize_lower_better(out["fid"])
    )
    lam = _safe_numeric(out.get("lambda_cond_adv"), 0.0)
    out["lambda_safety_score"] = np.where(
        lam <= 0,
        0.5,
        np.where(lam <= 0.0003, 1.0, np.where(lam <= 0.001, 0.7, 0.3)),
    )

    branches = out.get("round10_branch", pd.Series("", index=out.index)).astype(str)
    branch_counts = branches.value_counts()
    out["diversity_score"] = branches.map(lambda b: 1.0 / max(branch_counts.get(b, 1), 1))

    out["round10_cond_adv_score"] = (
        0.30 * out["conditional_leakage_improvement_score"]
        + 0.20 * out["cancer_retention_score"]
        + 0.15 * out["prototype_margin_score"]
        + 0.15 * out["global_alignment_safety_score"]
        + 0.10 * out["lambda_safety_score"]
        + 0.10 * out["diversity_score"]
    )
    out.loc[out["biology_collapse_risk"], "round10_cond_adv_score"] *= 0.25
    return out


def _model_id_col(df: pd.DataFrame) -> str:
    for col in ("ID", "model_id", "exp_id", "experiment_id"):
        if col in df.columns:
            return col
    raise ValueError("No model id column found")


def select_round10_cond_adv_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 24,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    force_baseline_models = force_baseline_models or []
    id_col = _model_id_col(aggregated_df)
    pool = annotate_round10_scores(aggregated_df)
    all_annotated = annotate_round10_scores(all_df) if not all_df.empty else pool

    nonzero_cond = pool[
        (_safe_numeric(pool.get("lambda_cond_adv"), 0.0) > 0)
        | pool.get("conditional_adv_enabled", False).fillna(False)
    ]
    if nonzero_cond.empty:
        raise ValueError(
            "No nonzero conditional ADV candidates in pool; fail fast before finetune."
        )

    selected_ids: List[str] = []
    group_map: Dict[str, str] = {}
    reason_map: Dict[str, str] = {}

    def _add(row: pd.Series, group: str, reason: str) -> bool:
        mid = str(row[id_col])
        if mid in selected_ids:
            return False
        if row.get("biology_collapse_risk", False) and group not in {"G9_forced_baseline", "G10_fill_ranked"}:
            return False
        selected_ids.append(mid)
        group_map[mid] = group
        reason_map[mid] = reason
        return True

    mask_10a = pool.get("round10_branch", "").astype(str).str.contains("10A", na=False)
    if mask_10a.any():
        best_10a = pool[mask_10a].sort_values("round10_cond_adv_score", ascending=False).iloc[0]
        _add(best_10a, "G9_forced_baseline", "10A_control_best_seed")

    for lam in LAMBDA_FORCE_VALUES:
        lam_rows = pool[_safe_numeric(pool.get("lambda_cond_adv"), 0.0) == lam]
        if lam_rows.empty:
            continue
        best = lam_rows.sort_values("round10_cond_adv_score", ascending=False).iloc[0]
        _add(best, "G7_lambda_diversity", f"best_lambda_{lam}")

    mask_10b = pool.get("round10_branch", "").astype(str).str.contains("10B", na=False)
    if mask_10b.any():
        best_10b = pool[mask_10b].sort_values("round10_cond_adv_score", ascending=False).iloc[0]
        _add(best_10b, "G5_best_10B_replacement", "best_10B_replacement")

    mask_10c = pool.get("round10_branch", "").astype(str).str.contains("10C", na=False)
    if mask_10c.any():
        best_10c = pool[mask_10c].sort_values("round10_cond_adv_score", ascending=False).iloc[0]
        _add(best_10c, "G6_best_10C_weak_global_guard", "best_10C_weak_global")

    for forced in force_baseline_models:
        forced_rows = all_annotated[all_annotated[id_col].astype(str).str.contains(forced, na=False)]
        if forced_rows.empty:
            forced_rows = pool[pool[id_col].astype(str).str.contains(forced, na=False)]
        if not forced_rows.empty:
            row = forced_rows.sort_values("round10_cond_adv_score", ascending=False).iloc[0]
            _add(row, "G9_forced_baseline", f"forced_{forced}")

    for group_name, strategy, _, quota in ROUND10_GROUP_SPECS:
        if len(selected_ids) >= top_k:
            break
        if strategy == "lowest_leakage":
            candidates = pool.sort_values("mean_conditional_leakage_strength", ascending=True)
        elif strategy == "leakage_improvement":
            candidates = pool.sort_values("conditional_leakage_improvement", ascending=False)
        elif strategy == "cancer_retention":
            candidates = pool.sort_values("kmeans_ari", ascending=False)
        elif strategy == "global_alignment":
            candidates = pool.sort_values("wasserstein", ascending=True)
        elif strategy == "best_10B":
            candidates = pool[mask_10b].sort_values("round10_cond_adv_score", ascending=False)
        elif strategy == "best_10C":
            candidates = pool[mask_10c].sort_values("round10_cond_adv_score", ascending=False)
        elif strategy == "lambda_diversity":
            candidates = pool.sort_values("round10_cond_adv_score", ascending=False)
        elif strategy == "dim_diversity":
            candidates = pool.sort_values("round10_cond_adv_score", ascending=False)
        elif strategy == "forced_baseline":
            continue
        else:
            candidates = pool.sort_values("round10_cond_adv_score", ascending=False)

        added = 0
        for _, row in candidates.iterrows():
            if added >= quota or len(selected_ids) >= top_k:
                break
            if _add(row, group_name, strategy):
                added += 1

    if len(selected_ids) < top_k:
        for _, row in pool.sort_values("round10_cond_adv_score", ascending=False).iterrows():
            if len(selected_ids) >= top_k:
                break
            _add(row, "G10_fill_ranked", "score_rank_fill")

    top_df = pool[pool[id_col].astype(str).isin(selected_ids)].copy()
    order = {mid: i for i, mid in enumerate(selected_ids)}
    top_df["_sel_order"] = top_df[id_col].astype(str).map(order)
    top_df = top_df.sort_values("_sel_order").drop(columns=["_sel_order"])
    top_df["round10_selection_group"] = top_df[id_col].astype(str).map(group_map)
    top_df["round10_selection_reason"] = top_df[id_col].astype(str).map(reason_map)

    info = {
        "selection_mode": "round10_cond_adv_qc",
        "top_k": top_k,
        "selected_count": len(top_df),
        "nonzero_cond_pool": len(nonzero_cond),
        "forced_baselines": force_baseline_models,
    }
    return top_df, info
