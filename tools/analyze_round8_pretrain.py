#!/usr/bin/env python3
"""Round 8 pretrain diagnostics for 8A control architecture and 8B VICReg architecture sweeps."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_round5_pretrain import _resolve, _stat, load_combined_tables
from tools.collapse_detection import annotate_alignment_collapse
from tools.round8_selection import annotate_round8_scores, encoder_family


def infer_branch(run_tag: str) -> str:
    tag = str(run_tag or "").lower()
    if "round8a" in tag or "control_arch" in tag:
        return "8A"
    if "round8b" in tag or "vicreg_arch" in tag:
        return "8B"
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
    probe = _stat(g.get("round8_downstream_probe_score"))
    vicreg = g[g.get("round8_vicreg_active", pd.Series(False, index=g.index)).fillna(False)]
    controls = g[g.get("round8_control_like", pd.Series(False, index=g.index)).fillna(False)]
    return {
        "branch": branch,
        "n": len(g),
        "mean_kmeans_ari": ari["mean"],
        "median_kmeans_ari": ari["median"],
        "mean_wasserstein": wass["mean"],
        "median_wasserstein": wass["median"],
        "mean_downstream_probe_score": probe["mean"],
        "best_vicreg_model": _best_id(vicreg, "round8_downstream_probe_score"),
        "best_control_model": _best_id(controls, "round8_downstream_probe_score"),
        "collapse_rate": float(g["alignment_collapse"].fillna(False).mean()),
        "structure_pass_rate": float(g["structure_pass"].fillna(False).mean()),
        "vicreg_active_rate": float(g.get("round8_vicreg_active", pd.Series(False, index=g.index)).fillna(False).mean()),
    }


def _group_summary(df: pd.DataFrame, group_col: str, value_col: str = "round8_downstream_probe_score") -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        probe = _stat(g.get(value_col))
        ari = _stat(g.get("kmeans_ari"))
        rows.append(
            {
                "group": key,
                "n": len(g),
                f"mean_{value_col}": probe["mean"],
                "mean_kmeans_ari": ari["mean"],
                "collapse_rate": float(g["alignment_collapse"].fillna(False).mean()),
                "best_model": _best_id(g, value_col),
            }
        )
    return pd.DataFrame(rows).sort_values("group")


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
    csv_path = os.path.join(out_dir, "round8_pretrain_diagnostics.csv")
    summary_df.to_csv(csv_path, index=False)

    detail_sections = []
    if not df.empty:
        work = df.copy()
        work["round8_encoder_family"] = work.apply(encoder_family, axis=1)
        for title, col in (
            ("Latent size", "round8_latent_size"),
            ("Encoder family", "round8_encoder_family"),
            ("Dropout rate", "dropout_rate"),
            ("lambda_cls", "lambda_cls"),
            ("gan_gen_update_interval", "gan_gen_update_interval"),
            ("gan_patience", "gan_patience"),
            ("VICReg var", "lambda_tumor_var"),
            ("VICReg cov", "lambda_tumor_cov"),
        ):
            if col in work.columns:
                gs = _group_summary(work, col)
                if not gs.empty:
                    detail_sections.append((title, gs))

    md_path = os.path.join(out_dir, "round8_pretrain_diagnostics.md")
    lines = [
        "# Round 8 Pretrain Diagnostics",
        "",
        f"- Total experiments: {len(df)}",
        "",
        "## Branch summaries (8A control / 8B VICReg)",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "_No data_",
        "",
    ]
    for title, gs in detail_sections:
        lines.extend([f"## By {title}", "", gs.to_markdown(index=False), ""])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"csv_path": csv_path, "md_path": md_path, "summary_df": summary_df}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Round 8 pretrain diagnostics")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", default="result/optimization_runs/round8_combined/reports")
    args = parser.parse_args(argv)

    df = load_combined_tables(args.run_dirs)
    if df.empty:
        print("No experiments found.")
        return 1
    df = annotate_alignment_collapse(df)
    df = annotate_round8_scores(df)
    df["branch"] = df.get("pretrain_run_tag", pd.Series("", index=df.index)).map(infer_branch)
    paths = write_reports(df, args.out_dir)
    print(f"Wrote {paths['csv_path']}")
    print(f"Wrote {paths['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
