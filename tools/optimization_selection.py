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


def load_all_pretrain_rows(result_dir: str) -> pd.DataFrame:
    from visualize_vaewc_results import load_experiment_data

    result_dir = _resolve_path(result_dir)
    exp_dirs = sorted(d for d in glob(os.path.join(result_dir, "exp_*")) if os.path.isdir(d))
    if not exp_dirs:
        return pd.DataFrame()
    return pd.DataFrame([load_experiment_data(d) for d in exp_dirs])


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


def select_top10_with_controls(
    aggregated_df: pd.DataFrame,
    n_ranked: int = 8,
    n_controls: int = 2,
) -> Tuple[pd.DataFrame, dict]:
    """Select best n_ranked + n_controls lambda_proto=0 controls from filtered pool."""
    if aggregated_df.empty:
        return aggregated_df.copy(), {
            "controls_available": 0,
            "controls_selected": 0,
            "shortage": True,
            "ranked_selected": 0,
            "total_selected": 0,
        }

    df = aggregated_df.sort_values("score_total", ascending=False, na_position="last").reset_index(drop=True)
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
    top10 = top10.sort_values("score_total", ascending=False, na_position="last").reset_index(drop=True)
    top10["selection_rank"] = range(1, len(top10) + 1)
    top10["is_control"] = top10["lambda_proto"].fillna(0.0) == 0.0

    info = {
        "controls_available": int(len(controls)),
        "controls_selected": int(len(selected_controls)),
        "shortage": len(selected_controls) < n_controls,
        "ranked_selected": int(len(selected_ranked)),
        "total_selected": int(len(top10)),
        "infonce_available": int(len(non_controls)),
    }
    return top10, info


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
) -> dict:
    selection_dir = _resolve_path(os.path.join(run_dir, "selection"))
    reports_dir = _resolve_path(os.path.join(run_dir, "reports"))
    os.makedirs(selection_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    all_df = load_all_pretrain_rows(result_dir)
    all_df = enrich_with_lambda_proto(all_df, _resolve_path(result_dir))
    all_path = os.path.join(selection_dir, "pretrain_all_candidates.csv")
    all_df.to_csv(all_path, index=False)

    filter_report_path = os.path.join(selection_dir, "filter_threshold_report.csv")
    if not all_df.empty and not no_filter:
        filter_report = build_filter_threshold_report(all_df, filter_config)
        filter_report.to_csv(filter_report_path, index=False)
    else:
        filter_report_path = ""

    run_visualize(result_dir, selection_dir, filter_config, select_top_k=20, no_filter=no_filter)

    aggregated_path = os.path.join(selection_dir, "aggregated_vaewc_results.csv")
    if not os.path.exists(aggregated_path):
        raise SelectionInsufficientError(
            "No experiments passed the quality filter; aggregated_vaewc_results.csv was not created."
        )

    aggregated_df = pd.read_csv(aggregated_path)
    aggregated_df = enrich_with_lambda_proto(aggregated_df, _resolve_path(result_dir))
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

    top10_df, info = select_top10_with_controls(aggregated_df)
    info["passing_total"] = len(aggregated_df)
    info["passing_controls"] = int((aggregated_df["lambda_proto"].fillna(0) == 0).sum())
    info["passing_infonce"] = int((aggregated_df["lambda_proto"].fillna(0) != 0).sum())
    info["sufficient_for_finetune"] = sufficient
    info["shortage_reason"] = shortage_reason.strip()
    info["filter_enabled"] = not no_filter
    info["min_passing_required"] = min_passing

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
        f"- All loaded experiments: {len(all_df)}",
        f"- Passed all thresholds: {len(aggregated_df)}",
        f"- Required for finetune: >= {min_passing} (controls >= {require_controls})",
        f"- Sufficient for finetune: {sufficient}",
    ]
    if shortage_reason:
        report_lines.append(f"- **Shortage:** {shortage_reason}")
    if filter_report_path:
        report_lines.append(f"- Threshold report: `selection/filter_threshold_report.csv`")
    report_lines.extend(
        [
            f"- Controls available (lambda_proto=0): {info['controls_available']}",
            f"- Controls selected: {info['controls_selected']}",
            f"- Ranked non-control slots filled: {info['ranked_selected']}",
            f"- Final Top-10 size: {info['total_selected']}",
        ]
    )
    if info["shortage"]:
        report_lines.append(
            "- Warning: fewer than two valid lambda_proto=0 controls were available after filtering."
        )
    report_lines.extend(["", "## Selected IDs (from filtered pool only)", ""])
    for _, row in top10_df.iterrows():
        control_tag = " [control]" if row.get("is_control") else ""
        report_lines.append(
            f"- {row['ID']}: score_total={row.get('score_total', 'NA')}{control_tag}"
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
