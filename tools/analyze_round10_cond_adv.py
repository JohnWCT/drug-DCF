#!/usr/bin/env python3
"""Analyze Round 10 Conditional ADV pretrain and downstream results."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import load_json, resolve_path

ROUND9_REPRO_BEST_AVG_TCGA = 0.5671
R7_ORIGINAL_EXP048_AVG_TCGA = 0.5918


def _read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.exists(resolve_path(path)):
        return pd.DataFrame()
    return pd.read_csv(resolve_path(path))


def _collect_pretrain_summaries(run_dir: str) -> pd.DataFrame:
    pretrain_dir = os.path.join(resolve_path(run_dir), "pretrain")
    rows = []
    if not os.path.isdir(pretrain_dir):
        return pd.DataFrame()
    for exp_name in sorted(os.listdir(pretrain_dir)):
        exp_path = os.path.join(pretrain_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        summary_path = os.path.join(exp_path, "run_summary.json")
        gan_path = os.path.join(exp_path, "gan_metrics.json")
        row = {"model_id": exp_name, "result_dir": exp_path}
        if os.path.exists(summary_path):
            payload = load_json(summary_path)
            params = payload.get("params", {})
            metrics = payload.get("metrics", {})
            row.update(params)
            row.update(metrics)
        if os.path.exists(gan_path):
            row.update(load_json(gan_path))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _compute_success_status(summary: pd.DataFrame, downstream: pd.DataFrame) -> str:
    if summary.empty:
        return "inconclusive"
    branch = summary.get("round10_branch", pd.Series("", index=summary.index)).astype(str)
    mask_10a = branch.str.contains("10A", na=False)
    mask_cond = branch.str.contains("10B|10C", na=False, regex=True)
    if not mask_10a.any() or not mask_cond.any():
        return "inconclusive"

    leak_col = "mean_conditional_leakage_strength"
    auc_col = "macro_conditional_domain_auc"
    leakage_10a = _safe_series(summary, leak_col).loc[mask_10a].mean()
    leakage_cond = _safe_series(summary, leak_col).loc[mask_cond].mean()
    if pd.isna(leakage_10a) or pd.isna(leakage_cond):
        leakage_10a = _safe_series(summary, auc_col).loc[mask_10a].mean()
        leakage_cond = _safe_series(summary, auc_col).loc[mask_cond].mean()
        if pd.isna(leakage_10a) or pd.isna(leakage_cond):
            # Pretrain-only: proxy via global alignment (lower wasserstein => better)
            wass_10a = _safe_series(summary, "wasserstein").loc[mask_10a].mean()
            wass_cond = _safe_series(summary, "wasserstein").loc[mask_cond].mean()
            if pd.notna(wass_10a) and pd.notna(wass_cond):
                improved = wass_cond <= wass_10a
            else:
                return "inconclusive"
        else:
            improved = leakage_cond < leakage_10a
    else:
        improved = leakage_cond < leakage_10a

    kmeans_cond = _safe_series(summary, "kmeans_ari").loc[mask_cond].mean()
    collapse = kmeans_cond < 0.30 if pd.notna(kmeans_cond) else False

    avg_tcga = np.nan
    if not downstream.empty and "Average_TCGA_AUC_mean" in downstream.columns:
        avg_tcga = pd.to_numeric(downstream["Average_TCGA_AUC_mean"], errors="coerce").max()

    if not improved:
        return "no_conditional_improvement"
    if collapse:
        return "unsafe_biology_collapse"
    if pd.notna(avg_tcga) and avg_tcga >= ROUND9_REPRO_BEST_AVG_TCGA * 0.95:
        return "success_conditional_and_downstream"
    if improved and not collapse:
        return "success_conditional_only"
    return "inconclusive"


def analyze_round10(
    run_dir: str,
    round9_diagnostics: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
    selection_path: Optional[str] = None,
) -> dict:
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    run_dir = resolve_path(run_dir)
    r9_dir = resolve_path(round9_diagnostics)

    summary = _collect_pretrain_summaries(run_dir)
    r9_model = _read_csv(os.path.join(r9_dir, "round9_model_level_summary.csv"))
    r9_by_cancer = _read_csv(
        os.path.join(r9_dir, "../reports/conditional_domain_auc_by_cancer.csv")
    )
    if r9_by_cancer.empty:
        r9_by_cancer = _read_csv(
            os.path.join(os.path.dirname(r9_dir), "reports/conditional_domain_auc_by_cancer.csv")
        )

    pretrain_path = os.path.join(outdir, "round10_cond_adv_pretrain_summary.csv")
    summary.to_csv(pretrain_path, index=False)

    vs_baseline_rows = []
    if not r9_model.empty and not summary.empty:
        r9_048 = r9_model[r9_model.get("model_id", r9_model.get("exp_id", "")).astype(str).str.contains("048")]
        baseline_leakage = pd.to_numeric(
            r9_048.get("mean_conditional_leakage_strength"), errors="coerce"
        ).mean()
        baseline_auc = pd.to_numeric(
            r9_048.get("macro_conditional_domain_auc"), errors="coerce"
        ).mean()
        branch_series = summary.get("round10_branch", pd.Series(dtype=str))
        for branch in branch_series.dropna().unique():
            sub = summary[summary["round10_branch"] == branch]
            sub_leak = _safe_series(sub, "mean_conditional_leakage_strength").mean()
            sub_auc = _safe_series(sub, "macro_conditional_domain_auc").mean()
            vs_baseline_rows.append(
                {
                    "round10_branch": branch,
                    "n_models": len(sub),
                    "mean_conditional_leakage": sub_leak,
                    "macro_conditional_domain_auc": sub_auc,
                    "delta_leakage_vs_r9_exp048": sub_leak - baseline_leakage
                    if pd.notna(sub_leak) and pd.notna(baseline_leakage)
                    else np.nan,
                    "delta_auc_vs_r9_exp048": sub_auc - baseline_auc
                    if pd.notna(sub_auc) and pd.notna(baseline_auc)
                    else np.nan,
                    "mean_kmeans_ari": _safe_series(sub, "kmeans_ari").mean(),
                    "mean_wasserstein": _safe_series(sub, "wasserstein").mean(),
                    "mean_fid": _safe_series(sub, "fid").mean(),
                    "mean_lambda_cond_adv": _safe_series(sub, "lambda_cond_adv").mean(),
                }
            )
    vs_df = pd.DataFrame(vs_baseline_rows)
    vs_path = os.path.join(outdir, "round10_cond_adv_vs_round9_baseline.csv")
    vs_df.to_csv(vs_path, index=False)

    by_cancer_path = os.path.join(outdir, "round10_cond_adv_by_cancer.csv")
    if not r9_by_cancer.empty:
        r9_by_cancer.to_csv(by_cancer_path, index=False)
    else:
        pd.DataFrame().to_csv(by_cancer_path, index=False)

    downstream = _read_csv(aggregate_path)
    downstream_path = os.path.join(outdir, "round10_downstream_summary.csv")
    if not downstream.empty:
        downstream.to_csv(downstream_path, index=False)
    else:
        pd.DataFrame().to_csv(downstream_path, index=False)

    status = _compute_success_status(summary, downstream)
    report_lines = [
        "# Round 10 Conditional ADV Report",
        "",
        f"**round10_success_status:** `{status}`",
        "",
        "## Q1. Conditional leakage",
        "",
    ]
    if not vs_df.empty:
        for _, row in vs_df.iterrows():
            report_lines.append(
                f"- {row['round10_branch']}: delta leakage vs R9 exp_048 = {row.get('delta_leakage_vs_r9_exp048', 'n/a')}"
            )
    report_lines.extend(
        [
            "",
            "## Downstream",
            "",
            f"- Round 9 reproduction best Avg TCGA: {ROUND9_REPRO_BEST_AVG_TCGA}",
            f"- R7 original exp_048 Avg TCGA: {R7_ORIGINAL_EXP048_AVG_TCGA}",
        ]
    )
    if not downstream.empty and "Average_TCGA_AUC_mean" in downstream.columns:
        best = pd.to_numeric(downstream["Average_TCGA_AUC_mean"], errors="coerce").max()
        report_lines.append(f"- Round 10 best Avg TCGA: {best}")

    report_path = os.path.join(outdir, "round10_final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return {
        "pretrain_summary": pretrain_path,
        "vs_round9": vs_path,
        "by_cancer": by_cancer_path,
        "downstream": downstream_path,
        "report": report_path,
        "round10_success_status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 10 Conditional ADV")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round9-diagnostics", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--selection", default=None)
    args = parser.parse_args()

    outputs = analyze_round10(
        run_dir=args.run_dir,
        round9_diagnostics=args.round9_diagnostics,
        outdir=args.outdir,
        aggregate_path=args.aggregate,
        selection_path=args.selection,
    )
    print(f"round10_success_status={outputs['round10_success_status']}")
    for key, path in outputs.items():
        if key != "round10_success_status":
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
