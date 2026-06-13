#!/usr/bin/env python3
"""Round 7 pretrain diagnostics for 7A control refinement and 7B VICReg ablation."""

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

from tools.analyze_round5_pretrain import _resolve, _stat, load_combined_tables
from tools.analyze_round6_pretrain import load_run_pretrain_table
from tools.collapse_detection import annotate_alignment_collapse
from tools.round7_selection import annotate_round7_scores


def infer_branch(run_tag: str) -> str:
    tag = str(run_tag or "").lower()
    if "round7a" in tag or "exp010_control" in tag:
        return "7A"
    if "round7b" in tag or "vicreg" in tag:
        return "7B"
    return "unknown"


def _best_id(frame: pd.DataFrame, sort_col: str, ascending: bool = False) -> Optional[str]:
    if frame.empty or sort_col not in frame.columns:
        return None
    row = frame.sort_values(sort_col, ascending=ascending, na_position="last").iloc[0]
    return str(row.get("ID"))


def _summarize_branch(g: pd.DataFrame, branch: str) -> dict:
    if g.empty:
        return {"branch": branch, "n": 0}
    ari = _stat(g.get("kmeans_ari"))
    wass = _stat(g.get("wasserstein"))
    fid = _stat(g.get("fid"))
    sweet = _stat(g.get("round7_sweetspot_score", g.get("sweetspot_score")))
    exp010_sim = _stat(g.get("round7_exp010_similarity_score"))
    non_collapse = g[~g["alignment_collapse"].fillna(False)]
    controls = g[g.get("round7_control_like", pd.Series(False, index=g.index)).fillna(False)]
    vicreg = g[g.get("round7_vicreg_active", pd.Series(False, index=g.index)).fillna(False)]
    return {
        "branch": branch,
        "n": len(g),
        "mean_kmeans_ari": ari["mean"],
        "median_kmeans_ari": ari["median"],
        "mean_wasserstein": wass["mean"],
        "median_wasserstein": wass["median"],
        "mean_fid": fid["mean"],
        "median_fid": fid["median"],
        "mean_exp010_similarity_score": exp010_sim["mean"],
        "mean_sweetspot_score": sweet["mean"],
        "best_exp010_similarity_model": _best_id(g, "round7_exp010_similarity_score"),
        "best_vicreg_model": _best_id(vicreg, "round7_downstream_probe_priority"),
        "best_control_model": _best_id(controls, "round7_exp010_similarity_score"),
        "best_noncollapse_model": _best_id(non_collapse, "round7_sweetspot_score"),
        "collapse_rate": float(g["alignment_collapse"].fillna(False).mean()),
        "structure_pass_rate": float(g["structure_pass"].fillna(False).mean()),
        "vicreg_active_rate": float(g.get("round7_vicreg_active", pd.Series(False, index=g.index)).fillna(False).mean()),
    }


def build_branch_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "branch" not in work.columns:
        work["branch"] = work.get("pretrain_run_tag", pd.Series("", index=work.index)).map(infer_branch)
    rows = []
    for branch in sorted(work["branch"].dropna().unique()):
        rows.append(_summarize_branch(work[work["branch"] == branch], branch))
    rows.append(_summarize_branch(work, "combined"))
    return pd.DataFrame(rows)


def write_reports(df: pd.DataFrame, out_dir: str) -> dict:
    out_dir = _resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_df = build_branch_summaries(df)
    csv_path = os.path.join(out_dir, "round7_pretrain_diagnostics.csv")
    summary_df.to_csv(csv_path, index=False)
    md_path = os.path.join(out_dir, "round7_pretrain_diagnostics.md")
    lines = [
        "# Round 7 Pretrain Diagnostics",
        "",
        f"- Total experiments: {len(df)}",
        "",
        "## Branch summaries (7A control / 7B VICReg)",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "_No data_",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"csv_path": csv_path, "md_path": md_path, "summary_df": summary_df}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Round 7 pretrain diagnostics")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", default="result/optimization_runs/round7_combined/reports")
    args = parser.parse_args(argv)

    df = load_combined_tables(args.run_dirs)
    if df.empty:
        print("No experiments found.")
        return 1
    df = annotate_alignment_collapse(df)
    df = annotate_round7_scores(df)
    df["branch"] = df.get("pretrain_run_tag", pd.Series("", index=df.index)).map(infer_branch)
    paths = write_reports(df, args.out_dir)
    print(f"Wrote {paths['csv_path']}")
    print(f"Wrote {paths['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
