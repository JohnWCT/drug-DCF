"""Round 8 architecture-diverse downstream-aware selection."""

from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from tools.round6_selection import annotate_sweetspot_scores, range_score
from tools.round7_selection import control_like_score, is_vicreg_active

KMEANS_IDEAL_LOW = 0.45
KMEANS_IDEAL_HIGH = 0.80
WASS_IDEAL_LOW = 0.45
WASS_IDEAL_HIGH = 0.75
KMEANS_COLLAPSE_CUTOFF = 0.30

ROUND8_OUTPUT_COLS = (
    "round8_selection_group",
    "round8_diversity_reason",
    "round8_vicreg_active",
    "round8_control_like",
    "round8_architecture_family",
    "round8_latent_size",
    "round8_encoder_family",
    "round8_downstream_probe_score",
    "round8_architecture_diversity_score",
    "round8_pretrain_rank",
)

ROUND8_GROUP_SPECS = (
    ("G1_vicreg_active_best", "vicreg_probe", False, 6),
    ("G2_control_best", "control_probe", False, 6),
    ("G3_latent64_vicreg", "latent64_vicreg", False, 4),
    ("G4_latent96_or_128_probe", "latent_large", False, 4),
    ("G5_low_dropout_probe", "low_dropout", False, 3),
    ("G6_high_encoder_capacity_probe", "high_encoder", False, 4),
    ("G7_moderate_wasserstein", "wasserstein_moderate", False, 3),
    ("G8_best_kmeans", "kmeans_ari", False, 3),
)

HIGH_ENCODER_FAMILIES = frozenset({"wide_768", "xl_1024"})


def _parse_encoder_dims(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def encoder_family(row: pd.Series) -> str:
    dims = _parse_encoder_dims(row.get("encoder_dims"))
    if not dims:
        return "unknown"
    if dims == [256, 128]:
        return "small_256"
    if dims == [384, 192]:
        return "mid_384"
    if dims == [512, 256, 128]:
        return "standard_512"
    if dims == [768, 384, 192]:
        return "wide_768"
    if dims == [1024, 512, 256]:
        return "xl_1024"
    return f"other_{dims[0]}"


def latent_probe_score(latent_size) -> float:
    ls = int(float(latent_size)) if pd.notna(latent_size) else 64
    return {
        64: 1.0,
        96: 0.8,
        48: 0.6,
        32: 0.5,
        128: 0.4,
    }.get(ls, 0.5)


def vicreg_priority_score(row: pd.Series) -> float:
    if is_vicreg_active(row):
        return 1.0
    if control_like_score(row) >= 0.99:
        return 0.6
    return 0.0


def kmeans_structure_score(row: pd.Series) -> float:
    ari = pd.to_numeric(row.get("kmeans_ari"), errors="coerce")
    return range_score(ari, KMEANS_IDEAL_LOW, KMEANS_IDEAL_HIGH)


def moderate_wasserstein_score(row: pd.Series) -> float:
    wass = pd.to_numeric(row.get("wasserstein"), errors="coerce")
    return range_score(wass, WASS_IDEAL_LOW, WASS_IDEAL_HIGH)


def architecture_family_label(row: pd.Series) -> str:
    ls = int(float(row.get("round8_latent_size", row.get("latent_size", 64)) or 64))
    enc = str(row.get("round8_encoder_family", encoder_family(row)))
    vicreg = "vicreg" if bool(row.get("round8_vicreg_active", is_vicreg_active(row))) else "control"
    return f"{vicreg}_lat{ls}_{enc}"


def _is_collapsed(row: pd.Series) -> bool:
    if bool(row.get("alignment_collapse", False)):
        return True
    ari = pd.to_numeric(row.get("kmeans_ari"), errors="coerce")
    return pd.notna(ari) and float(ari) < KMEANS_COLLAPSE_CUTOFF


def annotate_round8_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = annotate_sweetspot_scores(df)
    drop_cols = [c for c in ROUND8_OUTPUT_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out["round8_vicreg_active"] = out.apply(is_vicreg_active, axis=1)
    out["round8_control_like"] = out.apply(lambda r: control_like_score(r) >= 0.99, axis=1)
    out["round8_latent_size"] = pd.to_numeric(out.get("latent_size"), errors="coerce")
    out["round8_encoder_family"] = out.apply(encoder_family, axis=1)
    out["round8_architecture_family"] = out.apply(architecture_family_label, axis=1)

    sweet = pd.to_numeric(out.get("sweetspot_score"), errors="coerce").fillna(0.0)
    out["round8_downstream_probe_score"] = out.apply(
        lambda r: (
            0.25 * float(pd.to_numeric(r.get("sweetspot_score"), errors="coerce") or 0.0)
            + 0.20 * vicreg_priority_score(r)
            + 0.20 * latent_probe_score(r.get("latent_size"))
            + 0.15 * moderate_wasserstein_score(r)
            + 0.10 * kmeans_structure_score(r)
        ),
        axis=1,
    )
    out["round8_architecture_diversity_score"] = sweet * 0.0 + 1.0
    return out


def _rank_pool(pool: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    if pool.empty:
        return pool
    metric_map = {
        "vicreg_probe": "round8_downstream_probe_score",
        "control_probe": "round8_downstream_probe_score",
        "latent64_vicreg": "round8_downstream_probe_score",
        "latent_large": "round8_downstream_probe_score",
        "low_dropout": "round8_downstream_probe_score",
        "high_encoder": "round8_downstream_probe_score",
        "wasserstein_moderate": "round8_downstream_probe_score",
        "kmeans_ari": "kmeans_ari",
    }
    col = metric_map.get(metric, metric)
    if col not in pool.columns:
        return pool.head(0)
    ranked = pool.sort_values(col, ascending=ascending, na_position="last")
    if metric == "wasserstein_moderate":
        ranked = ranked.copy()
        ranked["_wmod"] = ranked.apply(moderate_wasserstein_score, axis=1)
        return ranked.sort_values("_wmod", ascending=False, na_position="last")
    if metric == "kmeans_ari":
        return ranked
    return ranked


def _group_eligible(row: pd.Series, group_name: str) -> bool:
    if group_name == "G1_vicreg_active_best":
        return bool(row.get("round8_vicreg_active", False))
    if group_name == "G2_control_best":
        return bool(row.get("round8_control_like", False))
    if group_name == "G3_latent64_vicreg":
        ls = pd.to_numeric(row.get("round8_latent_size", row.get("latent_size")), errors="coerce")
        return pd.notna(ls) and int(ls) == 64 and bool(row.get("round8_vicreg_active", False))
    if group_name == "G4_latent96_or_128_probe":
        ls = pd.to_numeric(row.get("round8_latent_size", row.get("latent_size")), errors="coerce")
        return pd.notna(ls) and int(ls) in (96, 128)
    if group_name == "G5_low_dropout_probe":
        dr = pd.to_numeric(row.get("dropout_rate"), errors="coerce")
        return pd.notna(dr) and float(dr) <= 0.05
    if group_name == "G6_high_encoder_capacity_probe":
        return str(row.get("round8_encoder_family", encoder_family(row))) in HIGH_ENCODER_FAMILIES
    return True


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
        if not _group_eligible(row, group_name):
            continue
        if _is_collapsed(row) and not bool(row.get("force_baseline", False)):
            continue
        rec = row.to_dict()
        rec["round8_selection_group"] = group_name
        rec["round8_diversity_reason"] = reason
        picks.append(rec)
        selected_ids.add(mid)
        if len(picks) >= limit:
            break
    return picks


def select_round8_architecture_broad_probe(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 50,
    force_baseline_models: Optional[list] = None,
    per_group_limits: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Architecture-diverse Top-K with downstream probe priority."""
    from tools.optimization_selection import _resolve_baseline_row

    force_baseline_models = force_baseline_models or []
    warnings: list = []
    annotated = annotate_round8_scores(aggregated_df)
    pretrain_rank = annotated.sort_values(
        "round8_downstream_probe_score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    pretrain_rank["round8_pretrain_rank"] = np.arange(1, len(pretrain_rank) + 1)
    rank_map = pretrain_rank.set_index("ID")["round8_pretrain_rank"].to_dict()
    pool = pretrain_rank.copy()

    selected_ids: set = set()
    selected_rows: list = []
    group_counts: dict = {}

    reason_map = {
        "G1_vicreg_active_best": "best active VICReg downstream probe",
        "G2_control_best": "best control-like downstream probe",
        "G3_latent64_vicreg": "latent=64 VICReg architecture probe",
        "G4_latent96_or_128_probe": "large latent architecture probe",
        "G5_low_dropout_probe": "low dropout architecture probe",
        "G6_high_encoder_capacity_probe": "high-capacity encoder probe",
        "G7_moderate_wasserstein": "moderate wasserstein alignment band",
        "G8_best_kmeans": "best kmeans structure retention",
    }

    for group_name, metric, ascending, default_limit in ROUND8_GROUP_SPECS:
        limit = per_group_limits.get(group_name, default_limit) if per_group_limits else default_limit
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
            rec["round8_pretrain_rank"] = rank_map.get(rec["ID"], np.nan)
            selected_rows.append(rec)

    for model_id in force_baseline_models:
        if model_id in selected_ids:
            continue
        row, w = _resolve_baseline_row(model_id, all_df)
        warnings.extend(w)
        row["round8_selection_group"] = "G9_forced_baseline"
        row["round8_diversity_reason"] = f"forced baseline {model_id}"
        row["round8_pretrain_rank"] = rank_map.get(model_id, np.nan)
        if model_id in rank_map:
            base = pool[pool["ID"].astype(str) == model_id]
            if not base.empty:
                for col in ROUND8_OUTPUT_COLS:
                    if col in base.columns and col not in row:
                        row[col] = base.iloc[0].get(col)
        if _is_collapsed(row) and model_id not in {"exp_746", "exp_001"}:
            warnings.append(f"forced baseline {model_id} may be collapsed; retained with warning")
        selected_rows.append(row)
        selected_ids.add(model_id)
    group_counts["G9_forced_baseline"] = sum(
        1 for r in selected_rows if r.get("round8_selection_group") == "G9_forced_baseline"
    )

    fill_ranked = pretrain_rank.sort_values("round8_downstream_probe_score", ascending=False)
    for _, row in fill_ranked.iterrows():
        if len(selected_rows) >= top_k:
            break
        mid = str(row["ID"])
        if mid in selected_ids:
            continue
        if _is_collapsed(row):
            continue
        rec = row.to_dict()
        rec["round8_selection_group"] = "G10_fill_ranked"
        rec["round8_diversity_reason"] = "downstream probe priority fill"
        rec["round8_pretrain_rank"] = rank_map.get(mid, np.nan)
        selected_rows.append(rec)
        selected_ids.add(mid)

    out = pd.DataFrame(selected_rows)
    if not out.empty:
        out = out.sort_values(
            ["round8_downstream_probe_score", "round8_vicreg_active"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        out["selection_rank"] = np.arange(1, len(out) + 1)
        if "lambda_proto" in out.columns:
            out["is_control"] = pd.to_numeric(out["lambda_proto"], errors="coerce").fillna(0.0) == 0.0
        else:
            out["is_control"] = out.get("round8_control_like", pd.Series(False, index=out.index)).fillna(False)

    controls_available = 0
    if "lambda_proto" in pretrain_rank.columns:
        lp = pd.to_numeric(pretrain_rank["lambda_proto"], errors="coerce").fillna(0.0)
        controls_available = int((lp == 0.0).sum())

    info = {
        "selection_mode": "round8_architecture_broad_probe",
        "total_selected": len(out),
        "top_k": top_k,
        "top_k_requested": top_k,
        "group_counts": group_counts,
        "force_baseline_models": force_baseline_models,
        "warnings": warnings,
        "ranking_primary_metric": "round8_downstream_probe_score",
        "ranking_secondary_metrics": [
            "round8_vicreg_active",
            "round8_latent_size",
            "round8_encoder_family",
        ],
        "controls_available": controls_available,
        "controls_selected": int(out["is_control"].fillna(False).sum()) if "is_control" in out.columns else 0,
        "ranked_selected": int(min(len(pretrain_rank), top_k)),
        "shortage": len(out) < top_k,
    }
    return out, info
