#!/usr/bin/env python3
"""Analyze Round 12 prototype alignment pretrain + downstream results."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import resolve_path, write_csv, write_md

ROUND11_BEST = 0.5828
R7_EXP048 = 0.5918
HIGH_PRIORITY_CANCERS = ["Brain", "Esophageal", "Liver", "Lung", "Ovarian"]


def _collect_pretrain_summaries(run_dir: str) -> pd.DataFrame:
    rows = []
    pretrain_dir = os.path.join(resolve_path(run_dir), "pretrain")
    for summary_path in sorted(glob.glob(os.path.join(pretrain_dir, "exp_*", "run_summary.json"))):
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        params = payload.get("params", {})
        exp_id = payload.get("exp_id", os.path.basename(os.path.dirname(summary_path)))
        row = {"model_id": exp_id, **metrics}
        for key in (
            "round12_branch",
            "source_anchor_proto_enabled",
            "lambda_proto_align",
            "proto_align_metric",
            "proto_align_start_epoch",
            "proto_align_full_epoch",
            "proto_ema_momentum",
            "reconstruction_loss_type",
            "smooth_l1_beta",
            "global_adv_mode",
            "lambda_cond_adv",
            "lambda_global_adv_multiplier",
            "mean_target_to_source_anchor_distance",
            "proto_align_loss_mean",
            "source_anchor_initialized_count",
        ):
            if key not in row and key in params:
                row[key] = params[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _load_baseline_summary(round11_root: str) -> pd.DataFrame:
    path = os.path.join(
        resolve_path(round11_root),
        "round12a_baseline_qc",
        "round11_top_prototype_gap_summary.csv",
    )
    alt = os.path.join(
        resolve_path(round11_root).replace("round11_stability_recon", "round12_proto_alignment"),
        "round12a_baseline_qc",
        "round11_top_prototype_gap_summary.csv",
    )
    for candidate in (path, alt):
        if os.path.isfile(candidate):
            return pd.read_csv(candidate)
    return pd.DataFrame()


def analyze_round12(
    run_dir: str,
    round11_root: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
    selection_path: Optional[str] = None,
    baseline_qc_path: Optional[str] = None,
) -> str:
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    pretrain_summary = _collect_pretrain_summaries(run_dir)
    write_csv(pretrain_summary, os.path.join(outdir, "round12_proto_pretrain_summary.csv"))

    baseline_df = _load_baseline_summary(round11_root)
    if baseline_qc_path and os.path.isfile(resolve_path(baseline_qc_path)):
        baseline_df = pd.read_csv(resolve_path(baseline_qc_path))

    exp035_proto = np.nan
    if not baseline_df.empty and "model_id" in baseline_df.columns:
        match = baseline_df[baseline_df["model_id"].astype(str) == "exp_035"]
        if not match.empty:
            exp035_proto = match.iloc[0].get("mean_same_cancer_proto_distance", np.nan)

    vs_rows = []
    if not pretrain_summary.empty:
        for _, row in pretrain_summary.iterrows():
            vs_rows.append(
                {
                    "model_id": row.get("model_id"),
                    "round12_branch": row.get("round12_branch"),
                    "lambda_proto_align": row.get("lambda_proto_align"),
                    "mean_target_to_source_anchor_distance": row.get(
                        "mean_target_to_source_anchor_distance", np.nan
                    ),
                    "exp_035_baseline_proto_distance": exp035_proto,
                    "proto_distance_delta_vs_exp035": (
                        float(row.get("mean_target_to_source_anchor_distance", np.nan))
                        - float(exp035_proto)
                        if pd.notna(row.get("mean_target_to_source_anchor_distance"))
                        and pd.notna(exp035_proto)
                        else np.nan
                    ),
                    "kmeans_ari": row.get("kmeans_ari"),
                    "wasserstein": row.get("wasserstein"),
                }
            )
    write_csv(pd.DataFrame(vs_rows), os.path.join(outdir, "round12_proto_alignment_vs_round11.csv"))

    downstream_df = pd.DataFrame()
    if aggregate_path and os.path.isfile(resolve_path(aggregate_path)):
        downstream_df = pd.read_csv(resolve_path(aggregate_path))
        write_csv(downstream_df, os.path.join(outdir, "round12_downstream_summary.csv"))

    if selection_path and os.path.isfile(resolve_path(selection_path)):
        write_csv(pd.read_csv(resolve_path(selection_path)), os.path.join(outdir, "round12_selection_summary.csv"))

    best_avg = np.nan
    best_model = ""
    if not downstream_df.empty and "Average_TCGA_AUC_mean" in downstream_df.columns:
        id_col = "ID" if "ID" in downstream_df.columns else "model_id"
        best_row = downstream_df.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]
        best_avg = float(best_row["Average_TCGA_AUC_mean"])
        best_model = str(best_row.get(id_col, ""))

    proto_improved = False
    if vs_rows:
        active = [r for r in vs_rows if float(r.get("lambda_proto_align") or 0) > 0]
        if active and pd.notna(exp035_proto):
            best_active = min(
                active,
                key=lambda r: float(r.get("mean_target_to_source_anchor_distance") or 1e9),
            )
            proto_improved = float(best_active.get("mean_target_to_source_anchor_distance", 1e9)) < float(
                exp035_proto
            )

    round13_go = (
        pd.notna(best_avg)
        and best_avg > ROUND11_BEST
        and proto_improved
    )

    lines = [
        "# Round 12 Final Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Pretrain models: {len(pretrain_summary)}",
        "",
        "## Downstream",
        "",
    ]
    if pd.notna(best_avg):
        lines.append(f"- Best model: **{best_model}** — Avg TCGA **{best_avg:.4f}**")
        lines.append(f"- vs Round 11 exp_035 ({ROUND11_BEST}): **{best_avg - ROUND11_BEST:+.4f}**")
        lines.append(f"- vs R7 exp_048 ({R7_EXP048}): **{best_avg - R7_EXP048:+.4f}**")
    else:
        lines.append("- Downstream aggregate not available (pretrain-only report).")

    lines.extend(
        [
            "",
            "## Prototype alignment",
            "",
            f"- exp_035 baseline prototype distance: {exp035_proto}",
            f"- Active proto configs reduced target→source anchor distance: **{proto_improved}**",
            "",
            "## Round 13 decision",
            "",
            f"**Recommendation:** `{'go_response_features' if round13_go else 'defer_round13'}`",
            "",
        ]
    )
    if round13_go:
        lines.append(
            "Prototype gap improved and downstream exceeded Round 11; proceed to "
            "prototype-distance response features in Step 2."
        )
    else:
        lines.append(
            "Consider Round 12.1: lower lambda_proto_align, later start epoch, "
            "or stronger weak global guard before response features."
        )

    report_path = os.path.join(outdir, "round12_final_report.md")
    write_md(report_path, lines)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 12 prototype alignment")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--round11-root",
        default="result/optimization_runs/round11_stability_recon",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--baseline-qc", default=None)
    args = parser.parse_args()

    report = analyze_round12(
        run_dir=args.run_dir,
        round11_root=args.round11_root,
        outdir=args.outdir,
        aggregate_path=args.aggregate,
        selection_path=args.selection,
        baseline_qc_path=args.baseline_qc,
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
