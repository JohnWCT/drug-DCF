"""Round 7 downstream-aware diverse selection."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from tools.round6_selection import (
    annotate_sweetspot_scores,
    range_score,
)

EXP010_KMEANS_LOW = 0.68
EXP010_KMEANS_HIGH = 0.80
EXP010_WASS_LOW = 0.55
EXP010_WASS_HIGH = 0.72

ROUND7_OUTPUT_COLS = (
    "round7_control_like",
    "round7_vicreg_active",
    "round7_exp010_similarity_score",
    "round7_sweetspot_score",
    "round7_integrated_proxy_score",
    "round7_downstream_probe_priority",
    "round7_selection_group",
    "round7_diversity_reason",
    "round7_pretrain_rank",
)

ROUND7_GROUP_SPECS = (
    ("G1_exp010_like_control", "exp010_similarity", False, 4),
    ("G2_vicreg_active", "vicreg_rank", False, 4),
    ("G3_best_sweetspot", "sweetspot_score", False, 4),
    ("G4_best_kmeans", "kmeans_ari", False, 3),
    ("G5_moderate_wasserstein", "wasserstein_moderate", False, 3),
    ("G6_high_integrated_proxy", "integrated_proxy", False, 3),
)


def _numeric_column(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def control_like_score(row: pd.Series) -> float:
    keys = (
        "lambda_tumor_topology",
        "lambda_class_gap",
        "lambda_tumor_supcon",
        "lambda_subspace_ortho",
    )
    for key in keys:
        val = row.get(key, row.get(f"final_gan_g_{key}_eff", 0))
        if pd.notna(val) and float(val) > 0:
            return 0.3
    lv = pd.to_numeric(row.get("lambda_tumor_var", row.get("final_gan_g_lambda_tumor_var_eff", 0)), errors="coerce")
    lc = pd.to_numeric(row.get("lambda_tumor_cov", row.get("final_gan_g_lambda_tumor_cov_eff", 0)), errors="coerce")
    if pd.notna(lv) and float(lv) > 0:
        return 0.7 if (pd.isna(lc) or float(lc) == 0) else 0.7
    if pd.notna(lc) and float(lc) > 0:
        return 0.7
    return 1.0


def is_vicreg_active(row: pd.Series) -> bool:
    lv = pd.to_numeric(row.get("lambda_tumor_var", row.get("final_gan_g_lambda_tumor_var_eff", 0)), errors="coerce")
    lc = pd.to_numeric(row.get("lambda_tumor_cov", row.get("final_gan_g_lambda_tumor_cov_eff", 0)), errors="coerce")
    if (pd.notna(lv) and float(lv) > 0) or (pd.notna(lc) and float(lc) > 0):
        for key in ("lambda_tumor_topology", "lambda_class_gap", "lambda_tumor_supcon", "lambda_subspace_ortho"):
            val = row.get(key, row.get(f"final_gan_g_{key}_eff", 0))
            if pd.notna(val) and float(val) > 0:
                return False
        return True
    return False


def latent_score_round7(latent_size) -> float:
    ls = int(float(latent_size)) if pd.notna(latent_size) else 64
    if ls == 64:
        return 1.0
    if ls == 32:
        return 0.5
    if ls == 128:
        return 0.0
    return 0.5


def compute_exp010_similarity_row(row: pd.Series) -> float:
    ari = pd.to_numeric(row.get("kmeans_ari"), errors="coerce")
    wass = pd.to_numeric(row.get("wasserstein"), errors="coerce")
    k = range_score(ari, EXP010_KMEANS_LOW, EXP010_KMEANS_HIGH)
    w = range_score(wass, EXP010_WASS_LOW, EXP010_WASS_HIGH)
    l = latent_score_round7(row.get("latent_size"))
    c = control_like_score(row)
    return 0.35 * k + 0.35 * w + 0.15 * l + 0.15 * c


def integrated_proxy_score(row: pd.Series) -> float:
    for col in ("Integrated_Average_TCGA_AUC_mean", "global_tcga_proxy", "Average_TCGA_AUC_mean"):
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            return float(np.clip(val, 0.0, 1.0))
    return float(row.get("sweetspot_tcga_proxy_score", 0.0) or 0.0)


def wasserstein_moderate_score(row: pd.Series) -> float:
    wass = pd.to_numeric(row.get("wasserstein"), errors="coerce")
    return range_score(wass, EXP010_WASS_LOW, EXP010_WASS_HIGH)


def annotate_round7_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = annotate_sweetspot_scores(df)
    drop_cols = [c for c in ROUND7_OUTPUT_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out["round7_control_like"] = out.apply(lambda r: control_like_score(r) >= 0.99, axis=1)
    out["round7_vicreg_active"] = out.apply(is_vicreg_active, axis=1)
    out["round7_exp010_similarity_score"] = out.apply(compute_exp010_similarity_row, axis=1)
    out["round7_sweetspot_score"] = pd.to_numeric(out.get("sweetspot_score"), errors="coerce").fillna(0.0)
    out["round7_integrated_proxy_score"] = out.apply(integrated_proxy_score, axis=1)
    out["round7_downstream_probe_priority"] = (
        0.40 * out["round7_exp010_similarity_score"]
        + 0.25 * out["round7_sweetspot_score"]
        + 0.20 * out["round7_integrated_proxy_score"]
        + 0.15 * out.apply(control_like_score, axis=1)
    )
    return out


def _rank_pool(pool: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    if pool.empty:
        return pool
    if metric == "exp010_similarity":
        col = "round7_exp010_similarity_score"
    elif metric == "vicreg_rank":
        col = "round7_downstream_probe_priority"
        pool = pool[pool["round7_vicreg_active"].fillna(False)]
    elif metric == "sweetspot_score":
        col = "round7_sweetspot_score"
    elif metric == "kmeans_ari":
        col = "kmeans_ari"
    elif metric == "wasserstein_moderate":
        col = "round7_exp010_similarity_score"
        pool = pool.copy()
        pool["_wmod"] = pool.apply(wasserstein_moderate_score, axis=1)
        col = "_wmod"
    elif metric == "integrated_proxy":
        col = "round7_integrated_proxy_score"
    else:
        col = metric
    if col not in pool.columns:
        return pool.head(0)
    return pool.sort_values(col, ascending=ascending, na_position="last")


def _pick_from_group(
    pool: pd.DataFrame,
    group_name: str,
    reason: str,
    metric: str,
    ascending: bool,
    limit: int,
    selected_ids: set,
) -> list:
    ranked = _rank_pool(pool, metric, ascending)
    picks = []
    for _, row in ranked.iterrows():
        mid = str(row["ID"])
        if mid in selected_ids:
            continue
        if group_name == "G1_exp010_like_control" and not bool(row.get("round7_control_like", False)):
            continue
        if group_name == "G2_vicreg_active" and not bool(row.get("round7_vicreg_active", False)):
            continue
        if bool(row.get("alignment_collapse", False)) and not bool(row.get("force_baseline", False)):
            continue
        rec = row.to_dict()
        rec["round7_selection_group"] = group_name
        rec["round7_diversity_reason"] = reason
        picks.append(rec)
        selected_ids.add(mid)
        if len(picks) >= limit:
            break
    return picks


def select_round7_diverse_downstream_probe(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 30,
    force_baseline_models: Optional[list] = None,
    per_group_limits: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Diverse Top-K covering control / VICReg / sweetspot / geometry buckets."""
    from tools.optimization_selection import _resolve_baseline_row

    force_baseline_models = force_baseline_models or []
    warnings = []
    annotated = annotate_round7_scores(aggregated_df)
    pretrain_rank = annotated.sort_values(
        "round7_downstream_probe_priority", ascending=False, na_position="last"
    ).reset_index(drop=True)
    pretrain_rank["round7_pretrain_rank"] = np.arange(1, len(pretrain_rank) + 1)
    rank_map = pretrain_rank.set_index("ID")["round7_pretrain_rank"].to_dict()
    pool = pretrain_rank.copy()

    selected_ids: set = set()
    selected_rows: list = []
    group_counts: dict = {}

    for group_name, metric, ascending, limit in ROUND7_GROUP_SPECS:
        reason_map = {
            "G1_exp010_like_control": "best exp010-like control neighborhood",
            "G2_vicreg_active": "best active VICReg candidate",
            "G3_best_sweetspot": "best sweetspot pretrain score",
            "G4_best_kmeans": "best kmeans structure retention",
            "G5_moderate_wasserstein": "best moderate wasserstein band",
            "G6_high_integrated_proxy": "best integrated downstream proxy",
        }
        picks = _pick_from_group(
            pool,
            group_name,
            reason_map.get(group_name, group_name),
            metric,
            ascending,
            limit,
            selected_ids,
        )
        group_counts[group_name] = len(picks)
        for rec in picks:
            rec["round7_pretrain_rank"] = rank_map.get(rec["ID"], np.nan)
            selected_rows.append(rec)

    for model_id in force_baseline_models:
        if model_id in selected_ids:
            continue
        row, w = _resolve_baseline_row(model_id, all_df)
        warnings.extend(w)
        row["round7_selection_group"] = "G7_historical_baseline"
        row["round7_diversity_reason"] = f"forced baseline {model_id}"
        row["round7_pretrain_rank"] = rank_map.get(model_id, np.nan)
        if model_id in rank_map:
            base = pool[pool["ID"].astype(str) == model_id]
            if not base.empty:
                for col in ROUND7_OUTPUT_COLS:
                    if col in base.columns and col not in row:
                        row[col] = base.iloc[0].get(col)
        selected_rows.append(row)
        selected_ids.add(model_id)
    group_counts["G7_historical_baseline"] = sum(
        1 for r in selected_rows if r.get("round7_selection_group") == "G7_historical_baseline"
    )

    fill_ranked = pretrain_rank.sort_values("round7_downstream_probe_priority", ascending=False)
    for _, row in fill_ranked.iterrows():
        if len(selected_rows) >= top_k:
            break
        mid = str(row["ID"])
        if mid in selected_ids:
            continue
        if bool(row.get("alignment_collapse", False)):
            continue
        rec = row.to_dict()
        rec["round7_selection_group"] = "G8_fill_ranked"
        rec["round7_diversity_reason"] = "downstream probe priority fill"
        rec["round7_pretrain_rank"] = rank_map.get(mid, np.nan)
        selected_rows.append(rec)
        selected_ids.add(mid)

    out = pd.DataFrame(selected_rows)
    if not out.empty:
        out = out.sort_values(
            ["round7_downstream_probe_priority", "round7_exp010_similarity_score"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        out["selection_rank"] = np.arange(1, len(out) + 1)
        if "lambda_proto" in out.columns:
            out["is_control"] = pd.to_numeric(out["lambda_proto"], errors="coerce").fillna(0.0) == 0.0
        else:
            out["is_control"] = out.get("round7_control_like", pd.Series(False, index=out.index)).fillna(False)

    controls_available = 0
    if "lambda_proto" in pretrain_rank.columns:
        lp = pd.to_numeric(pretrain_rank["lambda_proto"], errors="coerce").fillna(0.0)
        controls_available = int((lp == 0.0).sum())

    info = {
        "selection_mode": "round7_diverse_downstream_probe",
        "total_selected": len(out),
        "top_k": top_k,
        "top_k_requested": top_k,
        "group_counts": group_counts,
        "force_baseline_models": force_baseline_models,
        "warnings": warnings,
        "ranking_primary_metric": "round7_downstream_probe_priority",
        "ranking_secondary_metrics": [
            "round7_exp010_similarity_score",
            "round7_sweetspot_score",
            "round7_integrated_proxy_score",
        ],
        "controls_available": controls_available,
        "controls_selected": int(out["is_control"].fillna(False).sum()) if "is_control" in out.columns else 0,
        "ranked_selected": int(min(len(pretrain_rank), top_k)),
        "shortage": False,
    }
    return out, info
