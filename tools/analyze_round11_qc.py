#!/usr/bin/env python3
"""Analyze Round 11 stability + SmoothL1 reconstruction ablation."""

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

from tools.round9_diagnostics_common import load_json, resolve_path, write_csv, write_md

ROUND10_BEST = 0.5749
ROUND9_REPRO = 0.5671
R7_EXP048 = 0.5918


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
        row = {"model_id": exp_name, "result_dir": exp_path}
        for fname in ("run_summary.json", "gan_metrics.json"):
            fpath = os.path.join(exp_path, fname)
            if os.path.exists(fpath):
                payload = load_json(fpath)
                if fname == "run_summary.json":
                    row.update(payload.get("params", {}))
                    row.update(payload.get("metrics", {}))
                else:
                    row.update(payload)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_round11(
    run_dir: str,
    round10_root: str,
    round9_diagnostics: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
    selection_path: Optional[str] = None,
) -> str:
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    summary = _collect_pretrain_summaries(run_dir)
    downstream = _read_csv(aggregate_path or os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    selection = _read_csv(selection_path or os.path.join(run_dir, "selection", "pretrain_top10.csv"))

    write_csv(summary, os.path.join(outdir, "round11_model_level_summary.csv"))

    if not summary.empty and "reconstruction_loss_type" in summary.columns:
        recon = (
            summary.groupby("reconstruction_loss_type", dropna=False)
            .agg(
                n=("model_id", "count"),
                mean_wasserstein=("wasserstein", "mean"),
                mean_kmeans_ari=("kmeans_ari", "mean"),
                mean_fid=("fid", "mean"),
            )
            .reset_index()
        )
        write_csv(recon, os.path.join(outdir, "round11_reconstruction_ablation_summary.csv"))

    if not summary.empty and "round11_branch" in summary.columns:
        cond = (
            summary.groupby("round11_branch", dropna=False)
            .agg(
                n=("model_id", "count"),
                mean_lambda_cond_adv=("lambda_cond_adv", "mean"),
                mean_wasserstein=("wasserstein", "mean"),
                mean_kmeans_ari=("kmeans_ari", "mean"),
            )
            .reset_index()
        )
        write_csv(cond, os.path.join(outdir, "round11_condadv_stability_summary.csv"))

    if not downstream.empty:
        write_csv(downstream, os.path.join(outdir, "round11_downstream_summary.csv"))

    best_avg = np.nan
    if not downstream.empty and "Average_TCGA_AUC_mean" in downstream.columns:
        best_avg = downstream["Average_TCGA_AUC_mean"].max()

    round12_go = (
        pd.notna(best_avg)
        and best_avg >= ROUND10_BEST
        and not summary.empty
        and summary.get("mean_conditional_leakage_strength", pd.Series(dtype=float)).notna().any()
    )

    report_path = os.path.join(outdir, "round11_final_report.md")
    lines = [
        "# Round 11 Final Report",
        "",
        "## References",
        "",
        f"- Round 10 best (exp_111): {ROUND10_BEST}",
        f"- Round 9 reproduction: {ROUND9_REPRO}",
        f"- R7 exp_048: {R7_EXP048}",
        "",
        "## Pretrain summary",
        "",
        f"- Models with artifacts: {len(summary)}",
        f"- Selected for finetune: {len(selection)}",
        "",
        "## Downstream",
        "",
        f"- Best Average_TCGA_AUC_mean: {best_avg}",
        "",
        "## Round 12 decision",
        "",
        f"**Recommendation:** `{'go_prototype_alignment' if round12_go else 'defer_round12'}`",
        "",
        "Round 12 requires measured conditional leakage improvement, cancer retention,",
        "and downstream >= Round 10 exp_111.",
    ]
    write_md(report_path, lines)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 11 QC")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round10-root", default="result/optimization_runs/round10_cond_adv")
    parser.add_argument("--round9-diagnostics", default="result/optimization_runs/round9_diagnostics/final_report")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--selection", default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(args.run_dir, "final_report")
    path = analyze_round11(
        run_dir=args.run_dir,
        round10_root=args.round10_root,
        round9_diagnostics=args.round9_diagnostics,
        outdir=outdir,
        aggregate_path=args.aggregate,
        selection_path=args.selection,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
