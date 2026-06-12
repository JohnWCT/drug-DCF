"""Round 6 sweet-spot selection scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


IDEAL_KMEANS_LOW = 0.65
IDEAL_KMEANS_HIGH = 0.78
IDEAL_WASS_LOW = 0.55
IDEAL_WASS_HIGH = 0.70
PREFERRED_LATENT = 32
ACCEPTABLE_LATENT = 64
PENALIZED_LATENT = 128


def range_score(x: float, low: float, high: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    x = float(x)
    if low <= x <= high:
        return 1.0
    if x < low:
        return max(0.0, 1.0 - (low - x) / max(low, 1e-6))
    return max(0.0, 1.0 - (x - high) / max(high, 1e-6))


def latent_preference_score(latent_size) -> float:
    ls = int(float(latent_size)) if pd.notna(latent_size) else PREFERRED_LATENT
    if ls == PREFERRED_LATENT:
        return 1.0
    if ls == ACCEPTABLE_LATENT:
        return 0.7
    if ls == PENALIZED_LATENT:
        return 0.3
    return 0.5


def compute_sweetspot_score_row(row: pd.Series, tcga_proxy_weight: float = 0.15) -> dict:
    ari = pd.to_numeric(row.get("kmeans_ari"), errors="coerce")
    wass = pd.to_numeric(row.get("wasserstein"), errors="coerce")
    latent = row.get("latent_size", PREFERRED_LATENT)
    tcga_proxy = pd.to_numeric(
        row.get("global_tcga_proxy") or row.get("Average_TCGA_AUC_mean"),
        errors="coerce",
    )

    k_score = range_score(ari, IDEAL_KMEANS_LOW, IDEAL_KMEANS_HIGH)
    w_score = range_score(wass, IDEAL_WASS_LOW, IDEAL_WASS_HIGH)
    l_score = latent_preference_score(latent)

    if pd.notna(tcga_proxy):
        t_score = float(np.clip(tcga_proxy, 0.0, 1.0))
        w_k, w_w, w_t, w_l = 0.30, 0.25, tcga_proxy_weight, 0.10
        diversity_w = 1.0 - w_k - w_w - w_t - w_l
    else:
        t_score = 0.0
        w_k, w_w, w_t, w_l = 0.35, 0.30, 0.0, 0.10
        diversity_w = 0.25

    diversity_bonus = 0.0
    if bool(row.get("has_active_tumor_loss", False)):
        diversity_bonus = 0.5
    elif float(row.get("lambda_tumor_topology", 0) or 0) > 0:
        diversity_bonus = 0.3
    elif float(row.get("lambda_class_gap", 0) or 0) > 0:
        diversity_bonus = 0.2

    sweetspot = (
        w_k * k_score
        + w_w * w_score
        + w_t * t_score
        + w_l * l_score
        + diversity_w * diversity_bonus
    )
    sweetspot_pass = bool(
        k_score >= 0.5
        and not bool(row.get("alignment_collapse", False))
        and sweetspot >= 0.45
    )
    return {
        "sweetspot_kmeans_score": k_score,
        "sweetspot_wasserstein_score": w_score,
        "sweetspot_latent_score": l_score,
        "sweetspot_tcga_proxy_score": t_score,
        "sweetspot_diversity_bonus": diversity_bonus,
        "sweetspot_score": sweetspot,
        "sweetspot_pass": sweetspot_pass,
        "sweetspot_diversity_group": row.get("pretrain_run_tag", "unknown"),
    }


def annotate_sweetspot_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    lt = pd.to_numeric(out.get("lambda_tumor_topology"), errors="coerce").fillna(0.0)
    lg = pd.to_numeric(out.get("lambda_class_gap"), errors="coerce").fillna(0.0)
    ls = pd.to_numeric(out.get("lambda_tumor_supcon"), errors="coerce").fillna(0.0)
    lv = pd.to_numeric(out.get("lambda_tumor_var"), errors="coerce").fillna(0.0)
    lcov = pd.to_numeric(out.get("lambda_tumor_cov"), errors="coerce").fillna(0.0)
    lortho = pd.to_numeric(out.get("lambda_subspace_ortho"), errors="coerce").fillna(0.0)
    out["has_active_tumor_loss"] = (lt > 0) | (lg > 0) | (ls > 0) | (lv > 0) | (lcov > 0) | (lortho > 0)

    rows = []
    for _, row in out.iterrows():
        rows.append(compute_sweetspot_score_row(row))
    sweet = pd.DataFrame(rows, index=out.index)
    return pd.concat([out, sweet], axis=1)


def rank_round6_sweetspot(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by sweetspot score, then kmeans in ideal range, then moderate wasserstein."""
    annotated = annotate_sweetspot_scores(df)
    sort_cols = [
        ("sweetspot_score", False),
        ("sweetspot_kmeans_score", False),
        ("sweetspot_wasserstein_score", False),
        ("kmeans_ari", False),
        ("wasserstein", True),
        ("class_gap_loss", True),
    ]
    by, asc = [], []
    for col, direction in sort_cols:
        if col in annotated.columns:
            by.append(col)
            asc.append(direction)
    if not by:
        by, asc = ["sweetspot_score"], [False]
    return annotated.sort_values(by=by, ascending=asc, na_position="last").reset_index(drop=True)
