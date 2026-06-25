"""Round 16 focused brute-force downstream QC selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND16_OUTPUT_COLS = (
    "round16_selection_group",
    "round16_selection_reason",
    "round16_bruteforce_score",
    "mean_auc_across_seeds",
    "std_auc_across_seeds",
    "best_auc",
    "own_plus_delta_vs_none",
    "round16_model_key",
    "feature_mode",
    "combo_id",
    "prototype_feature_mode",
    "response_input_mode",
    "source_model_id",
)

FORCED_RETENTION = (
    ("r13_exp_008", "best_per_model"),
    ("r15c_exp_005", "best_per_model"),
    ("r15c_exp_024", "best_per_model"),
    ("r13_exp_035", "best_per_model"),
    ("none", "best_feature_mode"),
    ("own_plus_summary", "best_feature_mode"),
    ("own_plus_summary_no_l2", "best_feature_mode"),
    ("own_plus_summary_robust_scaler", "best_feature_mode"),
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


def _extract_round16_model_key(model_id: str) -> str:
    text = str(model_id)
    for key in ("r13_exp_008", "r15c_exp_005", "r15c_exp_024", "r13_exp_035"):
        if key in text:
            return key
    if "source_model_id" in text:
        return text
    return text.split("_none")[0].split("_own_plus")[0]


def _feature_mode_from_row(row: pd.Series) -> str:
    for col in ("feature_mode", "prototype_feature_mode", "feature_variant"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val)
    model_id = str(row.get("model_id", row.get("Model_ID", "")))
    for mode in (
        "own_plus_summary_robust_scaler",
        "own_plus_summary_no_initialized_flags",
        "own_plus_summary_no_gap",
        "own_plus_summary_no_l2",
        "own_plus_summary_zscore",
        "own_plus_summary",
        "none",
    ):
        if model_id.endswith(f"_{mode}") or f"_{mode}_" in model_id:
            return mode
    return "none"


def aggregate_seed_stats(all_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per model-feature-combo across seeds."""
    if all_df.empty:
        return pd.DataFrame()

    work = all_df.copy()
    metric = "Average_TCGA_AUC_mean"
    if metric not in work.columns:
        metric = "avg_tcga_auc_mean"
    if metric not in work.columns:
        return pd.DataFrame()

    work[metric] = _safe_numeric(work[metric])
    work["feature_mode"] = work.apply(_feature_mode_from_row, axis=1)
    work["round16_model_key"] = work.get("round16_model_key", work.get("source_model_id", "")).astype(str)
    if work["round16_model_key"].eq("").all() or work["round16_model_key"].eq("nan").all():
        id_col = _model_id_col(work) if any(c in work.columns for c in ("Model_ID", "ID", "model_id")) else None
        if id_col:
            work["round16_model_key"] = work[id_col].map(_extract_round16_model_key)

    group_cols = ["round16_model_key", "feature_mode", "combo_id"]
    for col in group_cols:
        if col not in work.columns:
            if col == "combo_id":
                work["combo_id"] = 0
            else:
                work[col] = "unknown"

    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        model_key, feature_mode, combo_id = keys
        vals = sub[metric].dropna()
        none_sub = work[
            (work["round16_model_key"] == model_key)
            & (work["feature_mode"] == "none")
            & (work["combo_id"] == combo_id)
        ]
        none_mean = _safe_numeric(none_sub[metric]).mean() if not none_sub.empty else np.nan
        mean_auc = float(vals.mean()) if not vals.empty else np.nan
        std_auc = float(vals.std()) if len(vals) > 1 else 0.0
        best_auc = float(vals.max()) if not vals.empty else np.nan
        delta = mean_auc - none_mean if pd.notna(none_mean) and pd.notna(mean_auc) else np.nan
        score = mean_auc - 0.25 * std_auc + 0.25 * max(delta if pd.notna(delta) else 0.0, 0.0)
        rows.append(
            {
                "round16_model_key": model_key,
                "feature_mode": feature_mode,
                "combo_id": int(combo_id),
                "model_id": str(sub[_model_id_col(sub)].iloc[0]) if len(sub) else "",
                "n_seeds": len(vals),
                "mean_auc_across_seeds": mean_auc,
                "std_auc_across_seeds": std_auc,
                "best_auc": best_auc,
                "own_plus_delta_vs_none": delta,
                "Global_TCGA_AUC_mean": _safe_numeric(sub.get("Global_TCGA_AUC_mean")).mean(),
                "round16_bruteforce_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("round16_bruteforce_score", ascending=False, na_position="last")


def annotate_round16_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ROUND16_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    if "mean_auc_across_seeds" not in out.columns and "Average_TCGA_AUC_mean" in out.columns:
        out["mean_auc_across_seeds"] = _safe_numeric(out["Average_TCGA_AUC_mean"])
    if "round16_bruteforce_score" not in out.columns or out["round16_bruteforce_score"].isna().all():
        mean_auc = _safe_numeric(out.get("mean_auc_across_seeds", out.get("Average_TCGA_AUC_mean")))
        std_auc = _safe_numeric(out.get("std_auc_across_seeds", 0), default=0.0)
        delta = _safe_numeric(out.get("own_plus_delta_vs_none", 0), default=0.0)
        out["round16_bruteforce_score"] = mean_auc - 0.25 * std_auc + 0.25 * np.maximum(delta, 0.0)
    return out


def select_round16_bruteforce_candidates(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 10,
    force_baseline_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    del force_baseline_models
    seed_summary = aggregate_seed_stats(all_df if not all_df.empty else aggregated_df)
    if seed_summary.empty:
        pool = annotate_round16_scores(aggregated_df)
        id_col = _model_id_col(pool)
        top = pool.sort_values("round16_bruteforce_score", ascending=False, na_position="last").head(top_k)
        return top, {"group_counts": {}, "seed_summary_rows": 0}

    selected_rows = []
    selected_keys = set()
    group_counts = {}

    def _add_row(row: pd.Series, group: str, reason: str) -> bool:
        key = (row["round16_model_key"], row["feature_mode"], int(row["combo_id"]))
        if key in selected_keys:
            return False
        selected_keys.add(key)
        enriched = row.to_dict()
        enriched["round16_selection_group"] = group
        enriched["round16_selection_reason"] = reason
        selected_rows.append(enriched)
        group_counts[group] = group_counts.get(group, 0) + 1
        return True

    for model_key, kind in FORCED_RETENTION:
        if kind == "best_per_model":
            sub = seed_summary[seed_summary["round16_model_key"] == model_key]
            if not sub.empty:
                row = sub.sort_values("round16_bruteforce_score", ascending=False, na_position="last").iloc[0]
                _add_row(row, f"forced_{model_key}", "best_per_source_model")
        else:
            sub = seed_summary[seed_summary["feature_mode"] == model_key]
            if not sub.empty:
                row = sub.sort_values("round16_bruteforce_score", ascending=False, na_position="last").iloc[0]
                _add_row(row, f"forced_{model_key}", "best_feature_variant")

    ranked = seed_summary.sort_values("round16_bruteforce_score", ascending=False, na_position="last")
    for _, row in ranked.iterrows():
        if len(selected_rows) >= top_k:
            break
        key = (row["round16_model_key"], row["feature_mode"], int(row["combo_id"]))
        if key in selected_keys:
            continue
        selected_keys.add(key)
        enriched = row.to_dict()
        enriched["round16_selection_group"] = "ranked_fill"
        enriched["round16_selection_reason"] = "composite_score_rank"
        selected_rows.append(enriched)
        group_counts["ranked_fill"] = group_counts.get("ranked_fill", 0) + 1

    top10_df = pd.DataFrame(selected_rows).head(top_k)
    info = {
        "group_counts": group_counts,
        "seed_summary_rows": len(seed_summary),
        "selection_mode": "round16_bruteforce_qc",
    }
    return top10_df, info
