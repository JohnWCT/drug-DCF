"""Control-aware Top-10 selection wrapping visualize_vaewc_results outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from glob import glob
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SELECTION_MODES = (
    "score_total",
    "round4_kmeans_first",
    "round4_weighted",
    "round4_1_structure_first",
    "round5_structure_first",
    "round6_sweetspot",
    "round7_diverse_downstream_probe",
    "round8_architecture_broad_probe",
)
STRUCTURE_FIRST_MODES = frozenset(
    {"round4_1_structure_first", "round5_structure_first", "round6_sweetspot"}
)
ROUND7_SELECTION_MODES = frozenset({"round7_diverse_downstream_probe"})
ROUND8_SELECTION_MODES = frozenset({"round8_architecture_broad_probe"})
MULTI_BRANCH_SELECTION_MODES = ROUND7_SELECTION_MODES | ROUND8_SELECTION_MODES
RANKING_PRIMARY_BY_MODE = {
    "score_total": "score_total",
    "round4_kmeans_first": "score_kmeans",
    "round4_weighted": "score_round4",
    "round4_1_structure_first": "wasserstein",
    "round5_structure_first": "wasserstein",
    "round6_sweetspot": "sweetspot_score",
    "round7_diverse_downstream_probe": "round7_downstream_probe_priority",
    "round8_architecture_broad_probe": "round8_downstream_probe_score",
}
RANKING_SECONDARY_BY_MODE = {
    "score_total": ["score_total"],
    "round4_kmeans_first": ["wasserstein", "fid", "mmd", "score_total"],
    "round4_weighted": ["score_round4", "score_kmeans", "wasserstein"],
    "round4_1_structure_first": ["kmeans_ari", "fid", "mmd"],
    "round5_structure_first": ["kmeans_ari", "fid", "mmd", "class_gap_loss"],
    "round6_sweetspot": ["sweetspot_kmeans_score", "sweetspot_wasserstein_score", "kmeans_ari", "wasserstein"],
    "round7_diverse_downstream_probe": [
        "round7_downstream_probe_priority",
        "round7_exp010_similarity_score",
        "round7_sweetspot_score",
    ],
    "round8_architecture_broad_probe": [
        "round8_downstream_probe_score",
        "round8_vicreg_active",
        "round8_latent_size",
    ],
}

DEFAULT_FORCE_BASELINE_PATHS = {
    "exp_746": "result/pretrain_vaewc/exp_746",
    "exp_018": "result/optimization_runs/vaewc_proto_infonce_round3_exp746/pretrain/exp_018",
    "exp_001": "result/optimization_runs/vaewc_round5_t2s_infonce_appendix/pretrain/exp_001",
    "exp_005": "result/optimization_runs/vaewc_round5_control_centered/pretrain/exp_005",
    "exp_010": "result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer/pretrain/exp_010",
    "exp_012": "result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer/pretrain/exp_012",
    "exp_048": "result/optimization_runs/vaewc_round7B_vicreg_focused_ablation/pretrain/exp_048",
    "exp_021": "result/optimization_runs/vaewc_round7B_vicreg_focused_ablation/pretrain/exp_021",
    "exp_188": "result/optimization_runs/vaewc_round8A_control_arch_broad/pretrain/exp_188",
}


class SelectionInsufficientError(RuntimeError):
    """Raised when too few experiments pass the quality filter for Top-10 finetune."""


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _read_lambda_proto_from_exp(result_dir: str, exp_id: str) -> float:
    params_path = os.path.join(result_dir, exp_id, "params.json")
    if not os.path.exists(params_path):
        return float("nan")
    with open(params_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", {})
    return float(params.get("lambda_proto", 0.0))


def enrich_with_lambda_proto(df: pd.DataFrame, result_dir: str) -> pd.DataFrame:
    out = df.copy()
    if "lambda_proto" not in out.columns:
        out["lambda_proto"] = out["ID"].map(lambda exp_id: _read_lambda_proto_from_exp(result_dir, exp_id))
    else:
        out["lambda_proto"] = pd.to_numeric(out["lambda_proto"], errors="coerce")
    return out


def load_all_pretrain_rows(result_dir: str, source_tag: str = "") -> pd.DataFrame:
    from visualize_vaewc_results import load_experiment_data

    result_dir = _resolve_path(result_dir)
    exp_dirs = sorted(d for d in glob(os.path.join(result_dir, "exp_*")) if os.path.isdir(d))
    if not exp_dirs:
        return pd.DataFrame()
    rows = []
    for d in exp_dirs:
        row = load_experiment_data(d)
        row["pretrain_result_dir"] = d
        row["result_folder"] = d
        if source_tag:
            row["pretrain_run_tag"] = source_tag
        rows.append(row)
    return pd.DataFrame(rows)


def merge_result_dir_paths(result_dir: str, result_dirs: Optional[list] = None) -> list[str]:
    """Primary result_dir plus optional extras (deduplicated, order preserved)."""
    merged = [_resolve_path(result_dir)]
    if result_dirs:
        for path in result_dirs:
            resolved = _resolve_path(path)
            if resolved not in merged:
                merged.append(resolved)
    return merged


def load_merged_pretrain_rows(result_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for result_dir in result_dirs:
        tag = os.path.basename(os.path.normpath(_resolve_path(result_dir)))
        df = load_all_pretrain_rows(result_dir, source_tag=tag)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_and_enrich_merged_results(merged_dirs: list[str]) -> pd.DataFrame:
    """Load pretrain rows and enrich each branch with its own result_dir metadata."""
    if len(merged_dirs) > 1:
        frames = []
        for d in merged_dirs:
            resolved_dir = _resolve_path(d)
            if os.path.basename(resolved_dir) == "pretrain":
                branch_tag = os.path.basename(os.path.dirname(resolved_dir))
            else:
                branch_tag = os.path.basename(resolved_dir)
            part = load_all_pretrain_rows(d, source_tag=branch_tag)
            if part.empty:
                continue
            part = enrich_selection_metadata(part, d)
            frames.append(part)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result_dir = merged_dirs[0]
    all_df = load_all_pretrain_rows(result_dir)
    if all_df.empty:
        return all_df
    return enrich_selection_metadata(all_df, result_dir)


def apply_structure_first_stage1_filter(all_df: pd.DataFrame, selection_mode: str) -> pd.DataFrame:
    """Structure-first hard filter; expects all_df already enriched per source result_dir."""
    if selection_mode == "round4_1_structure_first":
        from tools.collapse_detection import apply_round41_stage1_filter

        return apply_round41_stage1_filter(all_df)
    if selection_mode == "round5_structure_first":
        from tools.collapse_detection import apply_round5_stage1_filter

        return apply_round5_stage1_filter(all_df)
    if selection_mode == "round6_sweetspot":
        from tools.collapse_detection import apply_round6_stage1_filter

        return apply_round6_stage1_filter(all_df)
    if selection_mode in ROUND7_SELECTION_MODES | ROUND8_SELECTION_MODES:
        from tools.collapse_detection import apply_round6_stage1_filter

        return apply_round6_stage1_filter(all_df)
    raise ValueError(f"Not a structure-first selection mode: {selection_mode}")


def build_filter_threshold_report(
    all_df: pd.DataFrame,
    filter_config: str,
) -> pd.DataFrame:
    """Per-experiment pass/fail for each visualize_vaewc_filter threshold."""
    from visualize_vaewc_results import apply_quality_filter, load_filter_config

    filter_cfg = load_filter_config(_resolve_path(filter_config))
    thresholds = filter_cfg.get("thresholds", {})
    lower_better = set(filter_cfg.get("lower_is_better", []))
    higher_better = set(filter_cfg.get("higher_is_better", []))

    rows = []
    for _, row in all_df.iterrows():
        out = {"ID": row["ID"], "lambda_proto": row.get("lambda_proto", 0.0)}
        fail_cols = []
        for col, threshold in thresholds.items():
            val = pd.to_numeric(row.get(col), errors="coerce")
            if col in lower_better:
                ok = pd.notna(val) and float(val) <= float(threshold)
            elif col in higher_better:
                ok = pd.notna(val) and float(val) >= float(threshold)
            else:
                ok = True
            out[f"{col}_value"] = val
            out[f"{col}_pass"] = bool(ok)
            if not ok:
                fail_cols.append(col)
        out["pass_all_thresholds"] = len(fail_cols) == 0
        out["failed_metrics"] = ",".join(fail_cols)
        rows.append(out)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["pass_all_thresholds", "ID"], ascending=[False, True])
    return report


def _read_proto_guard_from_exp(result_dir: str, exp_id: str) -> dict:
    """Load proto checkpoint guard fields from gan_metrics.json or params.json."""
    out = {
        "proto_not_effective_checkpoint": False,
        "proto_effective_checkpoint_available": True,
        "proto_invalid": False,
        "proto_effective_epoch": None,
        "proto_start_epoch": None,
        "proto_full_epoch": None,
        "best_gan_epoch": None,
        "best_gan_epoch_overall": None,
        "best_gan_epoch_post_proto": None,
        "best_gan_loss_overall": None,
        "best_gan_loss_post_proto": None,
        "selection_checkpoint_type": "overall",
        "proto_mode": "combined",
        "proto_direction": "symmetric",
        "proto_detach": True,
        "lambda_cmmd": 0.0,
        "latent_size": None,
        "encoder_dims": None,
    }
    metrics_path = os.path.join(_resolve_path(result_dir), exp_id, "gan_metrics.json")
    params_path = os.path.join(_resolve_path(result_dir), exp_id, "params.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in out:
            if key in payload and payload[key] is not None:
                out[key] = payload[key]
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload.get("params", payload)
        for key in ("proto_mode", "proto_direction", "proto_detach", "lambda_cmmd", "latent_size", "encoder_dims"):
            if key in params:
                out[key] = params[key]
        if out["proto_start_epoch"] is None:
            out["proto_start_epoch"] = params.get("proto_start_epoch")
        if out["proto_full_epoch"] is None:
            out["proto_full_epoch"] = params.get("proto_full_epoch")
    lp = _read_lambda_proto_from_exp(result_dir, exp_id)
    if out.get("best_gan_epoch_overall") is None and out.get("best_gan_epoch") is not None:
        out["best_gan_epoch_overall"] = out["best_gan_epoch"]
    if out["proto_not_effective_checkpoint"] is False and out.get("best_gan_epoch_overall") is not None:
        pstart = out.get("proto_start_epoch")
        if lp > 0 and pstart is not None and int(out["best_gan_epoch_overall"]) < int(pstart):
            out["proto_not_effective_checkpoint"] = True
    if lp > 0:
        sel_type = str(out.get("selection_checkpoint_type", "none"))
        if sel_type == "post_proto":
            post_available = True
        else:
            from tools.pretrain_proto_schedule import post_proto_checkpoint_min_epoch

            post_epoch = int(out.get("best_gan_epoch_post_proto") or 0)
            min_post = out.get("post_proto_checkpoint_min_epoch")
            if min_post is None and out.get("proto_start_epoch") is not None:
                min_post = post_proto_checkpoint_min_epoch(
                    {
                        "proto_start_epoch": out["proto_start_epoch"],
                        "proto_full_epoch": out.get("proto_full_epoch", out["proto_start_epoch"]),
                    }
                )
            post_available = min_post is not None and post_epoch >= int(min_post)
        out["proto_effective_checkpoint_available"] = post_available
        out["proto_invalid"] = not post_available
    return out


def enrich_selection_metadata(df: pd.DataFrame, result_dir: str) -> pd.DataFrame:
    from tools.collapse_detection import annotate_alignment_collapse

    out = enrich_with_lambda_proto(df, result_dir)
    rows = []
    for _, row in out.iterrows():
        guard = _read_proto_guard_from_exp(result_dir, row["ID"])
        merged = {**row.to_dict(), **guard}
        rows.append(merged)
    enriched = pd.DataFrame(rows)
    return annotate_alignment_collapse(enriched)


def _rank_series(values: pd.Series, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", ascending=ascending, na_option="bottom")


def _build_score_deconfounding_rank(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("fid", "mmd", "wasserstein"):
        if col in df.columns:
            parts.append(_rank_series(df[col], ascending=True))
    if not parts:
        return pd.Series(0.0, index=df.index)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + part
    max_rank = combined.max()
    if max_rank <= 0:
        return pd.Series(0.0, index=df.index)
    return 1.0 - (combined - 1.0) / max_rank


def apply_selection_ranking(df: pd.DataFrame, selection_mode: str = "score_total") -> pd.DataFrame:
    """Sort candidates according to selection_mode."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"Unsupported selection_mode={selection_mode}. Use one of {SELECTION_MODES}.")

    out = df.copy()
    if selection_mode == "round4_weighted":
        score_kmeans = pd.to_numeric(out.get("score_kmeans"), errors="coerce").fillna(0.0)
        if "score_deconfounding" in out.columns:
            score_deconf = pd.to_numeric(out["score_deconfounding"], errors="coerce").fillna(0.0)
        else:
            score_deconf = _build_score_deconfounding_rank(out)
        out["score_round4"] = 0.6 * score_kmeans + 0.4 * score_deconf
        sort_cols = [
            ("score_round4", False),
            ("score_kmeans", False),
            ("wasserstein", True),
            ("fid", True),
            ("score_total", False),
        ]
    elif selection_mode == "round4_kmeans_first":
        sort_cols = [
            ("score_kmeans", False),
            ("wasserstein", True),
            ("fid", True),
            ("mmd", True),
            ("score_total", False),
        ]
    elif selection_mode == "round4_1_structure_first":
        from tools.collapse_detection import rank_round41_stage2

        return rank_round41_stage2(out)
    elif selection_mode == "round5_structure_first":
        from tools.collapse_detection import rank_round5_stage2

        return rank_round5_stage2(out)
    elif selection_mode == "round6_sweetspot":
        from tools.round6_selection import rank_round6_sweetspot

        return rank_round6_sweetspot(out)
    elif selection_mode == "round7_diverse_downstream_probe":
        from tools.round7_selection import annotate_round7_scores

        annotated = annotate_round7_scores(out)
        sort_cols = [
            ("round7_downstream_probe_priority", False),
            ("round7_exp010_similarity_score", False),
            ("round7_sweetspot_score", False),
            ("kmeans_ari", False),
        ]
        by = []
        ascending = []
        for col, direction in sort_cols:
            if col in annotated.columns:
                by.append(col)
                ascending.append(direction)
        if not by:
            by, ascending = ["round7_downstream_probe_priority"], [False]
        return annotated.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)
    elif selection_mode == "round8_architecture_broad_probe":
        from tools.round8_selection import annotate_round8_scores

        annotated = annotate_round8_scores(out)
        sort_cols = [
            ("round8_downstream_probe_score", False),
            ("round8_vicreg_active", False),
            ("kmeans_ari", False),
        ]
        by = []
        ascending = []
        for col, direction in sort_cols:
            if col in annotated.columns:
                by.append(col)
                ascending.append(direction)
        if not by:
            by, ascending = ["round8_downstream_probe_score"], [False]
        return annotated.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)
    else:
        sort_cols = [("score_total", False)]

    by = []
    ascending = []
    for col, asc in sort_cols:
        if col in out.columns:
            by.append(col)
            ascending.append(asc)
    if not by:
        by = ["score_total"]
        ascending = [False]

    return out.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)


def select_top10_with_controls(
    aggregated_df: pd.DataFrame,
    n_ranked: int = 8,
    n_controls: int = 2,
    selection_mode: str = "score_total",
) -> Tuple[pd.DataFrame, dict]:
    """Select best n_ranked + n_controls lambda_proto=0 controls from filtered pool."""
    if aggregated_df.empty:
        return aggregated_df.copy(), {
            "controls_available": 0,
            "controls_selected": 0,
            "shortage": True,
            "ranked_selected": 0,
            "total_selected": 0,
            "selection_mode": selection_mode,
            "ranking_primary_metric": RANKING_PRIMARY_BY_MODE.get(selection_mode, "score_total"),
            "ranking_secondary_metrics": RANKING_SECONDARY_BY_MODE.get(selection_mode, []),
        }

    df = apply_selection_ranking(aggregated_df, selection_mode=selection_mode)
    controls = df[df["lambda_proto"].fillna(0.0) == 0.0].copy()
    non_controls = df[df["lambda_proto"].fillna(0.0) != 0.0].copy()

    selected_controls = controls.head(n_controls)
    selected_ranked = non_controls.head(n_ranked)
    if len(selected_ranked) < n_ranked:
        remaining = df[~df["ID"].isin(pd.concat([selected_controls, selected_ranked])["ID"])]
        fill_count = n_ranked - len(selected_ranked)
        selected_ranked = pd.concat([selected_ranked, remaining.head(fill_count)], ignore_index=True)

    top10 = pd.concat([selected_ranked, selected_controls], ignore_index=True)
    top10 = top10.drop_duplicates(subset=["ID"], keep="first")
    top10 = apply_selection_ranking(top10, selection_mode=selection_mode)
    top10["selection_rank"] = range(1, len(top10) + 1)
    top10["is_control"] = top10["lambda_proto"].fillna(0.0) == 0.0

    info = {
        "controls_available": int(len(controls)),
        "controls_selected": int(len(selected_controls)),
        "shortage": len(selected_controls) < n_controls,
        "ranked_selected": int(len(selected_ranked)),
        "total_selected": int(len(top10)),
        "infonce_available": int(len(non_controls)),
        "selection_mode": selection_mode,
        "ranking_primary_metric": RANKING_PRIMARY_BY_MODE.get(selection_mode, "score_total"),
        "ranking_secondary_metrics": RANKING_SECONDARY_BY_MODE.get(selection_mode, []),
        "controls_selected_ids": selected_controls["ID"].tolist() if not selected_controls.empty else [],
    }
    return top10, info


def _resolve_baseline_row(model_id: str, all_df: pd.DataFrame) -> tuple[dict, list[str]]:
    warnings = []
    pretrain_path = DEFAULT_FORCE_BASELINE_PATHS.get(model_id)
    if pretrain_path is None:
        warnings.append(f"Unknown baseline model `{model_id}` (no default path).")
        return {"ID": model_id}, warnings
    abs_path = _resolve_path(pretrain_path)
    if not os.path.isdir(abs_path):
        warnings.append(f"Baseline checkpoint missing for `{model_id}`: {pretrain_path}")
    row = {"ID": model_id, "pretrain_result_dir": pretrain_path, "result_folder": pretrain_path}
    if not all_df.empty and model_id in set(all_df["ID"].astype(str)):
        row.update(all_df[all_df["ID"].astype(str) == model_id].iloc[0].to_dict())
    row["ID"] = model_id
    row["pretrain_result_dir"] = pretrain_path
    row["result_folder"] = pretrain_path
    row["is_control"] = True
    row["force_baseline"] = True
    return row, warnings


def select_top_k_with_baselines(
    aggregated_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 15,
    force_baseline_models: Optional[list] = None,
    selection_mode: str = "round5_structure_first",
    best_control_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Select top-K from pool, then force-append baselines and best control."""
    force_baseline_models = force_baseline_models or []
    warnings = []
    ranked = apply_selection_ranking(aggregated_df, selection_mode=selection_mode)
    selected_ids = set()
    selected_rows = []

    for _, row in ranked.iterrows():
        if len(selected_rows) >= top_k:
            break
        if row["ID"] in selected_ids:
            continue
        selected_rows.append(row.to_dict())
        selected_ids.add(row["ID"])

    controls = ranked[ranked["lambda_proto"].fillna(0.0) == 0.0]
    best_control_row = None
    if best_control_id and best_control_id in set(controls["ID"].astype(str)):
        best_control_row = controls[controls["ID"].astype(str) == best_control_id].iloc[0].to_dict()
    elif not controls.empty:
        best_control_row = controls.iloc[0].to_dict()
        best_control_id = str(best_control_row["ID"])

    for model_id in force_baseline_models:
        if model_id in selected_ids:
            continue
        row, w = _resolve_baseline_row(model_id, all_df)
        warnings.extend(w)
        selected_rows.append(row)
        selected_ids.add(model_id)

    if best_control_row and best_control_id not in selected_ids:
        selected_rows.append(best_control_row)
        selected_ids.add(best_control_id)

    out = pd.DataFrame(selected_rows)
    if out.empty:
        info = _round5_selection_info(
            out, ranked, top_k, force_baseline_models, best_control_id, warnings, selection_mode
        )
        return out, info

    out = apply_selection_ranking(out, selection_mode=selection_mode)
    out["selection_rank"] = range(1, len(out) + 1)
    out["is_control"] = out.get("lambda_proto", pd.Series(0, index=out.index)).fillna(0.0) == 0.0
    info = _round5_selection_info(
        out, ranked, top_k, force_baseline_models, best_control_id, warnings, selection_mode
    )
    return out, info


def _round5_selection_info(
    out: pd.DataFrame,
    ranked: pd.DataFrame,
    top_k: int,
    force_baseline_models: list,
    best_control_id: Optional[str],
    warnings: list,
    selection_mode: str,
) -> dict:
    controls_available = 0
    infonce_available = 0
    if not ranked.empty and "lambda_proto" in ranked.columns:
        lp = pd.to_numeric(ranked["lambda_proto"], errors="coerce").fillna(0.0)
        controls_available = int((lp == 0.0).sum())
        infonce_available = int((lp != 0.0).sum())
    controls_selected = int(out["is_control"].fillna(False).sum()) if "is_control" in out.columns else 0
    return {
        "total_selected": int(len(out)),
        "top_k_requested": top_k,
        "force_baseline_models": list(force_baseline_models),
        "best_control_id": best_control_id,
        "warnings": warnings,
        "selection_mode": selection_mode,
        "ranking_primary_metric": RANKING_PRIMARY_BY_MODE.get(selection_mode, "score_total"),
        "ranking_secondary_metrics": RANKING_SECONDARY_BY_MODE.get(selection_mode, []),
        "controls_available": controls_available,
        "controls_selected": controls_selected,
        "ranked_selected": int(min(len(ranked), top_k)) if not ranked.empty else 0,
        "shortage": False,
        "infonce_available": infonce_available,
    }


def run_visualize(
    result_dir: str,
    output_dir: str,
    filter_config: str,
    select_top_k: int = 20,
    no_filter: bool = False,
) -> None:
    cmd = [
        "python3",
        os.path.join(PROJECT_ROOT, "visualize_vaewc_results.py"),
        "--result_dir",
        _resolve_path(result_dir),
        "--output_dir",
        _resolve_path(output_dir),
        "--filter_config",
        _resolve_path(filter_config),
        "--select_top_k",
        str(select_top_k),
    ]
    if no_filter:
        cmd.append("--no_filter")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_model_select_from_top10(top10_df: pd.DataFrame) -> pd.DataFrame:
    from visualize_vaewc_results import build_finetune_model_select

    return build_finetune_model_select(top10_df, top_k=len(top10_df))


def write_selection_outputs(
    run_dir: str,
    result_dir: str,
    filter_config: str = "config/visualize_vaewc_filter.json",
    no_filter: bool = False,
    min_passing: int = 10,
    require_controls: int = 2,
    selection_mode: str = "score_total",
    exclude_proto_ineffective: bool = False,
    top_k: int = 10,
    force_baseline_models: Optional[list] = None,
    result_dirs: Optional[list] = None,
) -> dict:
    selection_dir = _resolve_path(os.path.join(run_dir, "selection"))
    reports_dir = _resolve_path(os.path.join(run_dir, "reports"))
    os.makedirs(selection_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    merged_dirs = merge_result_dir_paths(result_dir, result_dirs)
    all_df = load_and_enrich_merged_results(merged_dirs)
    all_path = os.path.join(selection_dir, "pretrain_all_candidates.csv")
    all_df.to_csv(all_path, index=False)

    filter_report_path = os.path.join(selection_dir, "filter_threshold_report.csv")
    if not all_df.empty and not no_filter:
        filter_report = build_filter_threshold_report(all_df, filter_config)
        filter_report.to_csv(filter_report_path, index=False)
    else:
        filter_report_path = ""

    aggregated_path = os.path.join(selection_dir, "aggregated_vaewc_results.csv")
    already_enriched_per_branch = (
        selection_mode in STRUCTURE_FIRST_MODES or selection_mode in MULTI_BRANCH_SELECTION_MODES
    )
    if already_enriched_per_branch:
        # all_df already enriched per branch; do not re-enrich with primary result_dir only.
        stage_mode = (
            "round6_sweetspot"
            if selection_mode in MULTI_BRANCH_SELECTION_MODES
            else selection_mode
        )
        aggregated_df = apply_structure_first_stage1_filter(all_df, stage_mode)
        aggregated_df.to_csv(aggregated_path, index=False)
    else:
        run_visualize(result_dir, selection_dir, filter_config, select_top_k=20, no_filter=no_filter)
        if not os.path.exists(aggregated_path):
            raise SelectionInsufficientError(
                "No experiments passed the quality filter; aggregated_vaewc_results.csv was not created."
            )

    aggregated_df = pd.read_csv(aggregated_path)
    if not already_enriched_per_branch:
        aggregated_df = enrich_selection_metadata(aggregated_df, _resolve_path(result_dir))

    excluded_proto = 0
    if exclude_proto_ineffective:
        before = len(aggregated_df)
        mask = pd.Series(True, index=aggregated_df.index)
        if "proto_not_effective_checkpoint" in aggregated_df.columns:
            mask = mask & (~aggregated_df["proto_not_effective_checkpoint"].fillna(False))
        if "proto_invalid" in aggregated_df.columns:
            lp = pd.to_numeric(aggregated_df.get("lambda_proto"), errors="coerce").fillna(0.0)
            mask = mask & ~((lp > 0) & aggregated_df["proto_invalid"].fillna(False))
        aggregated_df = aggregated_df[mask].copy()
        excluded_proto = before - len(aggregated_df)

    aggregated_df.to_csv(os.path.join(selection_dir, "pretrain_filtered_candidates.csv"), index=False)

    sufficient = True
    shortage_reason = ""
    if not no_filter:
        if len(aggregated_df) < min_passing:
            sufficient = False
            shortage_reason = (
                f"Only {len(aggregated_df)} experiments passed filter (need >= {min_passing}). "
                "Adjust InfoNCE sweep or pretrain_VAEwC.py and rerun pretrain."
            )
        controls_in_pool = int((aggregated_df["lambda_proto"].fillna(0) == 0).sum())
        if controls_in_pool < require_controls:
            sufficient = False
            shortage_reason += (
                f" Only {controls_in_pool} control(s) passed filter (need >= {require_controls})."
            )

    if selection_mode == "round7_diverse_downstream_probe":
        from tools.round7_selection import select_round7_diverse_downstream_probe

        top10_df, info = select_round7_diverse_downstream_probe(
            aggregated_df,
            all_df,
            top_k=top_k,
            force_baseline_models=force_baseline_models or [],
        )
    elif selection_mode == "round8_architecture_broad_probe":
        from tools.round8_selection import select_round8_architecture_broad_probe

        top10_df, info = select_round8_architecture_broad_probe(
            aggregated_df,
            all_df,
            top_k=top_k,
            force_baseline_models=force_baseline_models or [],
        )
    elif selection_mode in STRUCTURE_FIRST_MODES:
        top10_df, info = select_top_k_with_baselines(
            aggregated_df,
            all_df,
            top_k=top_k,
            force_baseline_models=force_baseline_models or [],
            selection_mode=selection_mode,
        )
    else:
        top10_df, info = select_top10_with_controls(aggregated_df, selection_mode=selection_mode)
    info["passing_total"] = len(aggregated_df)
    info["passing_controls"] = int((aggregated_df["lambda_proto"].fillna(0) == 0).sum())
    info["passing_infonce"] = int((aggregated_df["lambda_proto"].fillna(0) != 0).sum())
    info["sufficient_for_finetune"] = sufficient
    info["shortage_reason"] = shortage_reason.strip()
    info["filter_enabled"] = not no_filter
    info["min_passing_required"] = min_passing
    info["exclude_proto_ineffective"] = exclude_proto_ineffective
    info["excluded_proto_ineffective_count"] = excluded_proto
    info["top_k"] = top_k
    info["force_baseline_models"] = force_baseline_models or []
    if info.get("warnings"):
        report_warnings = list(info["warnings"])
    else:
        report_warnings = []

    top10_path = os.path.join(selection_dir, "pretrain_top10.csv")
    top10_df.to_csv(top10_path, index=False)

    model_select_path = os.path.join(selection_dir, "model_select.csv")
    if sufficient:
        model_select_df = build_model_select_from_top10(top10_df)
        model_select_df.to_csv(model_select_path, index=False)
    else:
        pd.DataFrame().to_csv(model_select_path, index=False)

    report_lines = [
        "# Pretrain Selection Report",
        "",
        f"- Filter enabled: {not no_filter}",
        f"- Selection mode: `{selection_mode}`",
        f"- Ranking primary: `{info.get('ranking_primary_metric', 'score_total')}`",
        f"- Ranking secondary: `{info.get('ranking_secondary_metrics', [])}`",
        f"- Exclude proto-ineffective checkpoints: {exclude_proto_ineffective}",
        f"- Excluded proto-ineffective count: {excluded_proto}",
        f"- All loaded experiments: {len(all_df)}",
        f"- Passed all thresholds: {len(aggregated_df)}",
        f"- Required for finetune: >= {min_passing} (controls >= {require_controls})",
        f"- Sufficient for finetune: {sufficient}",
    ]
    if shortage_reason:
        report_lines.append(f"- **Shortage:** {shortage_reason}")
    if report_warnings:
        report_lines.append("- **Baseline warnings:**")
        for w in report_warnings:
            report_lines.append(f"  - {w}")
    if filter_report_path:
        report_lines.append(f"- Threshold report: `selection/filter_threshold_report.csv`")
    report_lines.extend(
        [
            f"- Controls available (lambda_proto=0): {info.get('controls_available', 'NA')}",
            f"- Controls selected: {info.get('controls_selected', 'NA')}",
            f"- Ranked non-control slots filled: {info.get('ranked_selected', 'NA')}",
            f"- Final Top-10 size: {info.get('total_selected', len(top10_df))}",
        ]
    )
    if info.get("group_counts"):
        round_label = (
            "Round 8"
            if selection_mode == "round8_architecture_broad_probe"
            else "Round 7"
            if selection_mode == "round7_diverse_downstream_probe"
            else "Selection"
        )
        report_lines.extend(["", f"## {round_label} selection groups", ""])
        for group, count in info["group_counts"].items():
            report_lines.append(f"- {group}: {count}")
    if info.get("shortage"):
        report_lines.append(
            "- Warning: fewer than two valid lambda_proto=0 controls were available after filtering."
        )
    report_lines.extend(["", "## Selected IDs (from filtered pool only)", ""])
    for _, row in top10_df.iterrows():
        control_tag = " [control]" if row.get("is_control") else ""
        report_lines.append(
            f"- {row['ID']}: score_total={row.get('score_total', 'NA')} "
            f"score_kmeans={row.get('score_kmeans', 'NA')}{control_tag}"
        )
    report_path = os.path.join(reports_dir, "pretrain_selection_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    result = {
        "aggregated_path": aggregated_path,
        "all_candidates_path": all_path,
        "filter_report_path": filter_report_path,
        "top10_path": top10_path,
        "model_select_path": model_select_path,
        "report_path": report_path,
        "selection_info": info,
    }

    if not sufficient and not no_filter:
        raise SelectionInsufficientError(shortage_reason or "Insufficient passing candidates for Top-10 finetune.")

    return result
