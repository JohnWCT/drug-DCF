#!/usr/bin/env python3
"""Analyze Round 17 direct-prototype optimization results."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import load_json, resolve_path, write_csv

ROUND13_BEST = 0.6112395039184843
HISTORICAL_METRIC = "Average_TCGA_AUC"
INTEGRATED5_METRIC = "Integrated5_TargetMacro_TCGA_AUC"
INTEGRATED5_DRUG_METRIC = "Integrated5_DrugMacro_TCGA_AUC"

AACDR_PREFIXES = ("aacdr_tcga_only_", "aacdr_gdsc_intersect_")


def _feature_family(mode: str) -> str:
    mode = str(mode).lower()
    if mode == "none":
        return "z_only"
    if mode.startswith("own_plus_summary"):
        return "distance_summary"
    if "projected" in mode and "context" in mode:
        return "direct_projected_context"
    if "projected" in mode and "delta" in mode:
        return "direct_projected_delta"
    if mode == "minimal_source_only_min_margin":
        return "minimal_source"
    if "delta" in mode or mode.startswith("own_proto"):
        return "direct_proto"
    return "other"


def _read_aggregate(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.isfile(resolve_path(path)):
        return pd.DataFrame()
    return pd.read_csv(resolve_path(path))


def _parse_model_id_feature_mode(model_id: str) -> tuple[str, str]:
    model_id = str(model_id)
    match = re.match(r"^(r\d+c?_exp_\d+(?:_control)?)_(.+)$", model_id)
    if match:
        return match.group(1), match.group(2)
    return model_id, "unknown"


def _normalize_aggregate(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    out = agg.copy()
    if "Model_ID" in out.columns and "model_id" not in out.columns:
        out = out.rename(columns={"Model_ID": "model_id"})
    if "model_id" in out.columns and "feature_mode" not in out.columns:
        parsed = out["model_id"].map(_parse_model_id_feature_mode)
        out["round17_model_key"] = [p[0] for p in parsed]
        out["feature_mode"] = [p[1] for p in parsed]
    for base in (HISTORICAL_METRIC, INTEGRATED5_METRIC, INTEGRATED5_DRUG_METRIC):
        mean_col = f"{base}_mean"
        if mean_col in out.columns and base not in out.columns:
            out[base] = pd.to_numeric(out[mean_col], errors="coerce")
    return out


def _seed_summary(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    agg = _normalize_aggregate(agg)
    if f"{HISTORICAL_METRIC}_mean" in agg.columns and HISTORICAL_METRIC not in agg.columns:
        rows = []
        for _, row in agg.iterrows():
            entry = {
                "model_id": row.get("model_id"),
                "round17_model_key": row.get("round17_model_key"),
                "feature_mode": row.get("feature_mode"),
                "n_seeds": int(row.get("n_finetune_runs", 1)) if pd.notna(row.get("n_finetune_runs")) else 1,
            }
            for col in (HISTORICAL_METRIC, INTEGRATED5_METRIC, INTEGRATED5_DRUG_METRIC):
                mean_col = f"{col}_mean"
                std_col = f"{col}_std"
                if mean_col in agg.columns:
                    entry[f"{col}_mean"] = float(row[mean_col]) if pd.notna(row[mean_col]) else np.nan
                    entry[f"{col}_std"] = float(row[std_col]) if std_col in agg.columns and pd.notna(row.get(std_col)) else 0.0
                    entry[f"{col}_best"] = entry[f"{col}_mean"]
            rows.append(entry)
        return pd.DataFrame(rows)
    group_cols = [c for c in ("model_id", "feature_mode", "combo_id", "response_head_mode") if c in agg.columns]
    if not group_cols:
        group_cols = ["model_id"] if "model_id" in agg.columns else []
    metric_cols = [c for c in (HISTORICAL_METRIC, INTEGRATED5_METRIC, INTEGRATED5_DRUG_METRIC) if c in agg.columns]
    if not metric_cols or not group_cols:
        return pd.DataFrame()
    rows = []
    for keys, grp in agg.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        if "model_id" in row and "round17_model_key" not in row:
            parsed = _parse_model_id_feature_mode(row["model_id"])
            row["round17_model_key"] = parsed[0]
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
            row[f"{col}_best"] = float(vals.max()) if not vals.empty else np.nan
        row["n_seeds"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows)


def _five_target_summary(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    agg = _normalize_aggregate(agg)
    targets = {
        "gdsc_intersect13": HISTORICAL_METRIC,
        "tcga_only3": "tcga_only3_Average_TCGA_AUC",
        "dapl": "dapl_Average_TCGA_AUC",
        "aacdr_tcga_only": "aacdr_tcga_only_Average_TCGA_AUC",
        "aacdr_gdsc_intersect": "aacdr_gdsc_intersect_Average_TCGA_AUC",
    }
    rows = []
    for target, col in targets.items():
        value_col = col if col in agg.columns else f"{col}_mean"
        if value_col not in agg.columns:
            continue
        vals = pd.to_numeric(agg[value_col], errors="coerce").dropna()
        rows.append(
            {
                "eval_target": target,
                "metric_column": col,
                "mean": float(vals.mean()) if not vals.empty else np.nan,
                "best": float(vals.max()) if not vals.empty else np.nan,
                "n_rows": len(vals),
            }
        )
    return pd.DataFrame(rows)


def _ranking_table(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    hist_col = f"{HISTORICAL_METRIC}_mean"
    int5_col = f"{INTEGRATED5_METRIC}_mean"
    if hist_col not in seed_summary.columns or int5_col not in seed_summary.columns:
        return pd.DataFrame()
    work = seed_summary.copy()
    work["historical_rank"] = work[hist_col].rank(ascending=False, method="min")
    work["integrated5_rank"] = work[int5_col].rank(ascending=False, method="min")
    work["rank_delta_integrated5_minus_historical"] = work["integrated5_rank"] - work["historical_rank"]
    return work.sort_values(hist_col, ascending=False, na_position="last")


def _top_candidates(seed_summary: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    sort_col = f"{INTEGRATED5_METRIC}_mean"
    if sort_col not in seed_summary.columns:
        sort_col = f"{HISTORICAL_METRIC}_mean"
    return seed_summary.sort_values(sort_col, ascending=False, na_position="last").head(top_k).copy()


def _build_report(outdir: str, stage: str, seed_summary: pd.DataFrame, references: dict) -> str:
    r13 = float(references.get("round13_best", ROUND13_BEST))
    lines = [
        "# Round 17 Direct Prototype Report",
        "",
        f"## Stage {stage.upper()}",
        "",
        "### Key questions",
        "1. Do direct prototype features beat own_plus_summary on historical and Integrated5 metrics?",
        "2. Which projected delta/context dimension works best?",
        "3. Do AACDR targets change the ranking?",
        "",
    ]
    if not seed_summary.empty and f"{HISTORICAL_METRIC}_mean" in seed_summary.columns:
        best_hist = float(seed_summary[f"{HISTORICAL_METRIC}_mean"].max())
        best_i5 = float(seed_summary.get(f"{INTEGRATED5_METRIC}_mean", pd.Series([np.nan])).max())
        summary_row = seed_summary[seed_summary.get("feature_mode", pd.Series()) == "own_plus_summary"]
        summary_hist = float(summary_row[f"{HISTORICAL_METRIC}_mean"].max()) if not summary_row.empty else np.nan
        lines.extend(
            [
                f"- Best historical mean: **{best_hist:.4f}** (Round 13 ref {r13:.4f})",
                f"- Best Integrated5 mean: **{best_i5:.4f}**",
                f"- own_plus_summary historical mean: **{summary_hist:.4f}**" if pd.notna(summary_hist) else "- own_plus_summary: n/a",
                "",
            ]
        )
    report_path = os.path.join(outdir, "round17_final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def analyze_round17(
    run_dir: str,
    settings_path: str,
    aggregate_path: Optional[str],
    stage: str,
    outdir: str,
) -> dict:
    settings = load_json(settings_path)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    agg_path = aggregate_path or os.path.join(run_dir, "aggregate", "aggregate_scores.csv")
    agg = _normalize_aggregate(_read_aggregate(agg_path))

    seed_summary = _seed_summary(agg)
    if "feature_mode" in seed_summary.columns:
        seed_summary["feature_family"] = seed_summary["feature_mode"].map(_feature_family)

    feature_family = (
        seed_summary.groupby("feature_family", dropna=False)
        .agg(
            n_combos=("n_seeds", "sum"),
            mean_historical=(f"{HISTORICAL_METRIC}_mean", "mean"),
            best_historical=(f"{HISTORICAL_METRIC}_best", "max"),
            mean_integrated5=(f"{INTEGRATED5_METRIC}_mean", "mean"),
        )
        .reset_index()
        if not seed_summary.empty and "feature_family" in seed_summary.columns
        else pd.DataFrame()
    )

    head_family = pd.DataFrame()
    if "response_head_mode" in seed_summary.columns and not seed_summary.empty:
        head_family = (
            seed_summary.groupby("response_head_mode")
            .agg(mean_historical=(f"{HISTORICAL_METRIC}_mean", "mean"), mean_integrated5=(f"{INTEGRATED5_METRIC}_mean", "mean"))
            .reset_index()
        )

    outputs = {
        "round17_stage_summary": write_csv(seed_summary, os.path.join(outdir, f"round17_stage{stage}_summary.csv")),
        "round17_feature_family_summary": write_csv(
            feature_family, os.path.join(outdir, "round17_feature_family_summary.csv")
        ),
        "round17_head_family_summary": write_csv(head_family, os.path.join(outdir, "round17_head_family_summary.csv")),
        "round17_five_target_eval_summary": write_csv(
            _five_target_summary(agg), os.path.join(outdir, "round17_five_target_eval_summary.csv")
        ),
        "round17_historical_vs_integrated5_ranking": write_csv(
            _ranking_table(seed_summary), os.path.join(outdir, "round17_historical_vs_integrated5_ranking.csv")
        ),
        "round17_top_candidates": write_csv(
            _top_candidates(seed_summary), os.path.join(outdir, "round17_top_candidates.csv")
        ),
        "round17_tsne_artifact_index": write_csv(pd.DataFrame(), os.path.join(outdir, "round17_tsne_artifact_index.csv")),
        "round17_final_report": _build_report(outdir, stage, seed_summary, settings.get("references", {})),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 17 direct-prototype results")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--settings", default="config/round17_direct_proto_settings.json")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--stage", default="17a")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outputs = analyze_round17(args.run_dir, args.settings, args.aggregate, args.stage, args.outdir)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
