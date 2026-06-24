#!/usr/bin/env python3
"""Analyze Round 15 reproducibility + exp_008 route rescue results."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import resolve_path, write_csv

ROUND13_BEST = 0.6112395039184843
ROUND14_BEST = 0.5908654300736795
TARGET_FLOOR = 0.6000


def _read_aggregate(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.isfile(resolve_path(path)):
        return pd.DataFrame()
    return pd.read_csv(resolve_path(path))


def _collect_pretrain_summaries(run_dir: str) -> pd.DataFrame:
    rows = []
    pretrain_dir = os.path.join(resolve_path(run_dir), "pretrain")
    for summary_path in sorted(glob.glob(os.path.join(pretrain_dir, "exp_*", "run_summary.json"))):
        exp_dir = os.path.dirname(summary_path)
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        params = payload.get("params", {})
        exp_id = payload.get("exp_id", os.path.basename(exp_dir))
        row = {"model_id": exp_id, **metrics}
        for key in (
            "round15_branch",
            "route_id",
            "source_model",
            "lambda_tumor_var",
            "lambda_tumor_cov",
            "tumor_vicreg_start_epoch",
            "tumor_vicreg_full_epoch",
            "kmeans_ari",
            "random_seed",
        ):
            if key not in row and key in params:
                row[key] = params[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_col(df: pd.DataFrame) -> str:
    for col in ("Average_TCGA_AUC_mean", "avg_tcga_auc_mean"):
        if col in df.columns:
            return col
    return "Average_TCGA_AUC_mean"


def _id_col(df: pd.DataFrame) -> str:
    for col in ("Model_ID", "model_id", "ID"):
        if col in df.columns:
            return col
    return "Model_ID"


def _seed_stability_summary(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    work = agg.copy()
    idc = _id_col(work)
    metric = _metric_col(work)
    if metric not in work.columns:
        return pd.DataFrame()

    rows = []
    mask15a = work[idc].astype(str).str.contains("r15a_exp_008", na=False)
    for feature_mode in ("none", "own_plus_summary"):
        sub = work[mask15a & work[idc].astype(str).str.contains(feature_mode, na=False)]
        if sub.empty:
            continue
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        rows.append(
            {
                "branch": "15A",
                "feature_mode": feature_mode,
                "n_seeds": len(sub),
                "mean_avg_tcga": float(vals.mean()) if not vals.empty else np.nan,
                "std_avg_tcga": float(vals.std()) if len(vals) > 1 else 0.0,
                "min_avg_tcga": float(vals.min()) if not vals.empty else np.nan,
                "max_avg_tcga": float(vals.max()) if not vals.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _z_vs_summary_delta(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    work = agg.copy()
    idc = _id_col(work)
    metric = _metric_col(work)
    if metric not in work.columns:
        return pd.DataFrame()

    rows = []
    for branch in ("a", "b", "c"):
        prefix = f"r15{branch}_"
        branch_df = work[work[idc].astype(str).str.startswith(prefix)]
        if branch_df.empty:
            continue
        for model_id in sorted(branch_df["source_model_id"].dropna().unique()) if "source_model_id" in branch_df.columns else []:
            none_rows = branch_df[
                (branch_df.get("prototype_feature_mode", "") == "none")
                | branch_df[idc].astype(str).str.endswith("_none")
                | branch_df[idc].astype(str).str.contains(f"{model_id}_none")
            ]
            sum_rows = branch_df[
                (branch_df.get("prototype_feature_mode", "") == "own_plus_summary")
                | branch_df[idc].astype(str).str.contains("own_plus_summary")
            ]
            none_val = pd.to_numeric(none_rows[metric], errors="coerce").max()
            sum_val = pd.to_numeric(sum_rows[metric], errors="coerce").max()
            if pd.isna(none_val) and pd.isna(sum_val):
                continue
            rows.append(
                {
                    "round15_branch": f"15{branch.upper()}",
                    "source_model_id": model_id,
                    "avg_tcga_none": none_val,
                    "avg_tcga_own_plus_summary": sum_val,
                    "delta_own_plus_summary_minus_none": (
                        float(sum_val - none_val) if pd.notna(none_val) and pd.notna(sum_val) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _exp008_route_summary(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    work = agg.copy()
    idc = _id_col(work)
    metric = _metric_col(work)
    mask = work[idc].astype(str).str.contains("exp_008", na=False)
    sub = work[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [idc, metric]
    for extra in ("Global_TCGA_AUC_mean", "prototype_feature_mode", "round15_branch", "random_seed"):
        if extra in sub.columns:
            cols.append(extra)
    return sub[cols].sort_values(metric, ascending=False, na_position="last")


def _vicreg_rescue_summary(pretrain_df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    if pretrain_df.empty:
        return pd.DataFrame()
    work = pretrain_df.copy()
    work["lambda_tumor_var"] = pd.to_numeric(work.get("lambda_tumor_var"), errors="coerce")
    work["lambda_tumor_cov"] = pd.to_numeric(work.get("lambda_tumor_cov"), errors="coerce")
    grouped = (
        work.groupby(["lambda_tumor_var", "lambda_tumor_cov", "tumor_vicreg_start_epoch"], dropna=False)
        .agg(n_models=("model_id", "count"), mean_kmeans_ari=("kmeans_ari", "mean"))
        .reset_index()
    )
    if not agg.empty:
        idc = _id_col(agg)
        metric = _metric_col(agg)
        c_rows = agg[agg[idc].astype(str).str.startswith("r15c_")]
        if not c_rows.empty and metric in c_rows.columns:
            grouped["best_downstream_avg_tcga"] = pd.to_numeric(c_rows[metric], errors="coerce").max()
    return grouped


def _build_final_report_md(agg: pd.DataFrame, seed_df: pd.DataFrame, exp008_df: pd.DataFrame) -> str:
    lines = [
        "# Round 15 Repro + exp_008 Route Rescue — Final Report",
        "",
        f"- Round 13 best reference: **{ROUND13_BEST:.4f}**",
        f"- Round 14 best reference: **{ROUND14_BEST:.4f}**",
        "",
        "## Q1. Round 13 best 5-seed reproducibility?",
    ]
    if not seed_df.empty:
        summary_row = seed_df[seed_df["feature_mode"] == "own_plus_summary"]
        if not summary_row.empty:
            mean_val = summary_row.iloc[0]["mean_avg_tcga"]
            std_val = summary_row.iloc[0]["std_avg_tcga"]
            lines.append(f"- 15A own_plus_summary mean ± std: **{mean_val:.4f} ± {std_val:.4f}**")
            lines.append(f"- Target floor {TARGET_FLOOR:.4f}: {'PASS' if mean_val >= TARGET_FLOOR else 'FAIL'}")
    else:
        lines.append("- Seed stability summary not available (partial run).")

    lines.extend(["", "## exp_008 route downstream"])
    if not exp008_df.empty:
        metric = _metric_col(exp008_df)
        best = exp008_df.iloc[0]
        lines.append(f"- Best exp_008 route: **{best[metric]:.4f}**")
    else:
        lines.append("- Pending.")

    lines.extend(["", "## Stack comparison"])
    if not agg.empty:
        idc = _id_col(agg)
        metric = _metric_col(agg)
        top = agg.sort_values(metric, ascending=False, na_position="last").head(10)
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            val = row.get(metric, np.nan)
            lines.append(f"{i}. `{row[idc]}` — {val:.4f}" if pd.notna(val) else f"{i}. `{row[idc]}` — NA")

    go = False
    if not seed_df.empty:
        srow = seed_df[seed_df["feature_mode"] == "own_plus_summary"]
        if not srow.empty:
            go = float(srow.iloc[0]["mean_avg_tcga"]) >= TARGET_FLOOR
    lines.extend(
        [
            "",
            "## Round 16 Go / No-Go",
            f"- **{'GO' if go else 'NO-GO'}** — "
            f"{'importance-aware weighting' if go else 'final validation / stability analysis'}",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_round15(
    run_dir: str,
    round13_root: str,
    round14_root: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
) -> dict:
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    agg = _read_aggregate(aggregate_path or os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    pretrain_df = _collect_pretrain_summaries(run_dir)
    manifest_path = os.path.join(run_dir, "manifests", "finetune_dispatch_manifest.csv")
    ft_manifest = pd.read_csv(manifest_path) if os.path.isfile(manifest_path) else pd.DataFrame()

    seed_df = _seed_stability_summary(agg)
    delta_df = _z_vs_summary_delta(agg)
    exp008_df = _exp008_route_summary(agg)
    vicreg_df = _vicreg_rescue_summary(pretrain_df, agg)

    repro_summary = pd.DataFrame(
        [
            {
                "round13_best_reference": ROUND13_BEST,
                "round14_best_reference": ROUND14_BEST,
                "n_finetune_jobs_manifest": len(ft_manifest),
                "n_aggregate_rows": len(agg),
                "n_pretrain_rows": len(pretrain_df),
            }
        ]
    )

    paths = {
        "round15_repro_summary.csv": repro_summary,
        "round15_exp008_route_summary.csv": exp008_df,
        "round15_vicreg_rescue_summary.csv": vicreg_df,
        "round15_z_vs_own_plus_summary_delta.csv": delta_df,
        "round15_seed_stability_summary.csv": seed_df,
    }
    for name, df in paths.items():
        write_csv(df, os.path.join(outdir, name))

    report_md = _build_final_report_md(agg, seed_df, exp008_df)
    report_path = os.path.join(outdir, "round15_final_report.md")
    os.makedirs(outdir, exist_ok=True)
    with open(resolve_path(report_path), "w", encoding="utf-8") as f:
        f.write(report_md)

    output_paths = {k: os.path.join(outdir, k) for k in paths}
    output_paths["round15_final_report.md"] = report_path
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 15 repro + rescue")
    parser.add_argument("--run-dir", default="result/optimization_runs/round15_repro_rescue")
    parser.add_argument("--round13-root", default="result/optimization_runs/round13_proto_response")
    parser.add_argument("--round14-root", default="result/optimization_runs/round14_vicreg_stabilizer")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(args.run_dir, "reports")
    outputs = analyze_round15(
        args.run_dir,
        args.round13_root,
        args.round14_root,
        outdir,
        aggregate_path=args.aggregate,
    )
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
