"""Alignment collapse detection for pretrain / selection (Round 4.1)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Round 4 empirical references (control mean / exp_746)
DEFAULT_CONTROL_MEAN_KMEANS_ARI = 0.703
DEFAULT_EXP746_KMEANS_ARI = 0.679
DEFAULT_STRUCTURE_ABSOLUTE_MIN = 0.65
DEFAULT_STRUCTURE_RELATIVE_RATIO = 0.90
DEFAULT_WASSERSTEIN_COLLAPSE_MAX = 0.50
DEFAULT_KMEANS_COLLAPSE_MAX = 0.30


def structure_pass_mask(
    kmeans_ari: pd.Series,
    control_mean_kmeans_ari: float = DEFAULT_CONTROL_MEAN_KMEANS_ARI,
    absolute_min: float = DEFAULT_STRUCTURE_ABSOLUTE_MIN,
    relative_ratio: float = DEFAULT_STRUCTURE_RELATIVE_RATIO,
) -> pd.Series:
    ari = pd.to_numeric(kmeans_ari, errors="coerce")
    relative_min = float(control_mean_kmeans_ari) * float(relative_ratio)
    threshold = max(float(absolute_min), relative_min)
    return ari >= threshold


def deconfounding_relaxed_pass_mask(
    wasserstein: pd.Series,
    wasserstein_max: float = 0.70,
) -> pd.Series:
    wass = pd.to_numeric(wasserstein, errors="coerce")
    return wass <= float(wasserstein_max)


def annotate_alignment_collapse(
    df: pd.DataFrame,
    control_mean_kmeans_ari: float = DEFAULT_CONTROL_MEAN_KMEANS_ARI,
    exp746_kmeans_ari: float = DEFAULT_EXP746_KMEANS_ARI,
    structure_absolute_min: float = DEFAULT_STRUCTURE_ABSOLUTE_MIN,
    wasserstein_collapse_max: float = DEFAULT_WASSERSTEIN_COLLAPSE_MAX,
    kmeans_collapse_max: float = DEFAULT_KMEANS_COLLAPSE_MAX,
) -> pd.DataFrame:
    """Add collapse / structure / deconfounding pass columns to candidate table."""
    out = df.copy()
    ari = pd.to_numeric(out.get("kmeans_ari"), errors="coerce")
    wass = pd.to_numeric(out.get("wasserstein"), errors="coerce")

    out["kmeans_ari_baseline_ref"] = float(control_mean_kmeans_ari)
    out["kmeans_ari_ratio_to_baseline"] = ari / float(control_mean_kmeans_ari)
    out["structure_pass"] = structure_pass_mask(
        ari,
        control_mean_kmeans_ari=control_mean_kmeans_ari,
        absolute_min=structure_absolute_min,
    )
    out["deconfounding_pass"] = deconfounding_relaxed_pass_mask(wass)

    wass_good = wass <= float(wasserstein_collapse_max)
    ari_bad = ari < float(kmeans_collapse_max)
    out["wasserstein_improved_but_structure_collapsed"] = wass_good & ari_bad

    collapse = pd.Series(False, index=out.index)
    reasons = pd.Series("", index=out.index, dtype=object)

    mask_global = out["wasserstein_improved_but_structure_collapsed"].fillna(False)
    collapse = collapse | mask_global
    reasons = reasons.mask(mask_global, "global_alignment_destroyed_tumor_structure")

    if "alignment_collapse" in out.columns:
        collapse = collapse | out["alignment_collapse"].fillna(False)

    out["alignment_collapse"] = collapse
    out["collapse_reason"] = reasons.replace("", np.nan)

    if "proto_effective_checkpoint_available" in out.columns:
        lp = pd.to_numeric(out.get("lambda_proto"), errors="coerce").fillna(0.0)
        proto_invalid = (lp > 0) & (~out["proto_effective_checkpoint_available"].fillna(True))
        out.loc[proto_invalid, "alignment_collapse"] = True
        out.loc[proto_invalid, "collapse_reason"] = out.loc[proto_invalid, "collapse_reason"].fillna(
            "proto_invalid_no_post_proto_checkpoint"
        )

    out["exp746_kmeans_ari_ref"] = float(exp746_kmeans_ari)
    return out


def apply_round41_stage1_filter(
    df: pd.DataFrame,
    wasserstein_max: float = 0.70,
    control_mean_kmeans_ari: float = DEFAULT_CONTROL_MEAN_KMEANS_ARI,
    structure_absolute_min: float = DEFAULT_STRUCTURE_ABSOLUTE_MIN,
    exclude_collapse: bool = True,
    exclude_proto_invalid: bool = True,
) -> pd.DataFrame:
    """Hard filter: structure + relaxed deconfounding; optional collapse / proto invalid exclusion."""
    annotated = annotate_alignment_collapse(
        df,
        control_mean_kmeans_ari=control_mean_kmeans_ari,
        structure_absolute_min=structure_absolute_min,
    )
    mask = annotated["structure_pass"].fillna(False) & annotated["deconfounding_pass"].fillna(False)
    if exclude_collapse:
        mask = mask & (~annotated["alignment_collapse"].fillna(False))
    if exclude_proto_invalid and "proto_effective_checkpoint_available" in annotated.columns:
        lp = pd.to_numeric(annotated.get("lambda_proto"), errors="coerce").fillna(0.0)
        proto_ok = (lp <= 0) | annotated["proto_effective_checkpoint_available"].fillna(False)
        mask = mask & proto_ok
    return annotated[mask].copy()


def rank_round41_stage2(df: pd.DataFrame) -> pd.DataFrame:
    """Rank survivors: wasserstein ↑, kmeans_ari ↓, fid ↑, mmd ↑."""
    sort_cols = []
    for col, asc in [
        ("wasserstein", True),
        ("kmeans_ari", False),
        ("fid", True),
        ("mmd", True),
        ("score_total", False),
    ]:
        if col in df.columns:
            sort_cols.append((col, asc))
    if not sort_cols:
        return df.copy()
    by = [c for c, _ in sort_cols]
    asc = [a for _, a in sort_cols]
    return df.sort_values(by=by, ascending=asc, na_position="last").reset_index(drop=True)
