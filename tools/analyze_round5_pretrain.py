#!/usr/bin/env python3
"""Round 5 pretrain diagnostics across control / class-gap / t2s appendix branches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.collapse_detection import annotate_alignment_collapse
from tools.optimization_selection import enrich_selection_metadata


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_run_pretrain_table(result_dir: str, run_tag: str = "") -> pd.DataFrame:
    result_dir = _resolve(result_dir)
    rows = []
    for metrics_path in sorted(glob(os.path.join(result_dir, "exp_*", "gan_metrics.json"))):
        exp_id = os.path.basename(os.path.dirname(metrics_path))
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["ID"] = exp_id
        if run_tag:
            payload["pretrain_run_tag"] = run_tag
        params_path = os.path.join(os.path.dirname(metrics_path), "params.json")
        if os.path.exists(params_path):
            with open(params_path, "r", encoding="utf-8") as f:
                p = json.load(f).get("params", {})
            for k in (
                "lambda_proto",
                "lambda_class_gap",
                "class_gap_metric",
                "latent_size",
                "encoder_dims",
                "proto_direction",
            ):
                if k not in payload and k in p:
                    payload[k] = p[k]
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return enrich_selection_metadata(df, result_dir)


def load_combined_tables(run_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        pretrain_dir = _resolve(run_dir)
        if not os.path.isdir(os.path.join(pretrain_dir, "exp_proto_000")) and os.path.isdir(
            os.path.join(pretrain_dir, "pretrain")
        ):
            pretrain_dir = os.path.join(pretrain_dir, "pretrain")
        tag = os.path.basename(os.path.normpath(_resolve(run_dir)))
        part = load_run_pretrain_table(pretrain_dir, run_tag=tag)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _stat(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    return {
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "median": float(s.median()) if s.notna().any() else np.nan,
    }


def _summarize_group(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {"n": 0}
    ari = _stat(g.get("kmeans_ari"))
    fid = _stat(g.get("fid"))
    wass = _stat(g.get("wasserstein"))
    mmd = _stat(g.get("mmd"))
    collapse_rate = float(g["alignment_collapse"].fillna(False).mean())
    structure_rate = float(g["structure_pass"].fillna(False).mean())
    proto_invalid_rate = float(g["proto_invalid"].fillna(False).mean()) if "proto_invalid" in g.columns else 0.0

    def _best_id_from(frame: pd.DataFrame, sort_col: str, ascending: bool) -> Optional[str]:
        if sort_col not in frame.columns or frame.empty:
            return None
        row = frame.sort_values(sort_col, ascending=ascending, na_position="last").iloc[0]
        return str(row.get("ID"))

    non_collapse = g[~g["alignment_collapse"].fillna(False)]
    return {
        "n": n,
        "mean_kmeans_ari": ari["mean"],
        "median_kmeans_ari": ari["median"],
        "mean_fid": fid["mean"],
        "median_fid": fid["median"],
        "mean_wasserstein": wass["mean"],
        "median_wasserstein": wass["median"],
        "mean_mmd": mmd["mean"],
        "collapse_rate": collapse_rate,
        "structure_pass_rate": structure_rate,
        "proto_invalid_rate": proto_invalid_rate,
        "filter_pass_rate": structure_rate,
        "best_model_by_kmeans": _best_id_from(g, "kmeans_ari", False),
        "best_model_by_wasserstein": _best_id_from(g, "wasserstein", True),
        "best_noncollapse_model": _best_id_from(non_collapse, "wasserstein", True),
        "best_downstream_model": None,
    }


def build_group_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summaries = []

    def add(name: str, mask: pd.Series):
        summaries.append({"group": name, **_summarize_group(df[mask])})

    lp = pd.to_numeric(df.get("lambda_proto"), errors="coerce").fillna(0.0)
    lg = pd.to_numeric(df.get("lambda_class_gap"), errors="coerce").fillna(0.0)
    add("control_lambda_proto_0_class_gap_0", (lp == 0) & (lg == 0))
    add("class_gap_branch", lg > 0)
    add("t2s_infonce_appendix", lp > 0)
    if "pretrain_run_tag" in df.columns:
        for tag in sorted(df["pretrain_run_tag"].dropna().unique()):
            add(f"run_{tag}", df["pretrain_run_tag"] == tag)
    if "latent_size" in df.columns:
        for ls in sorted(pd.to_numeric(df["latent_size"], errors="coerce").dropna().unique()):
            add(f"latent_size_{int(ls)}", pd.to_numeric(df["latent_size"], errors="coerce") == ls)
    if "class_gap_metric" in df.columns:
        for metric in sorted(df["class_gap_metric"].dropna().unique()):
            add(f"class_gap_metric_{metric}", df["class_gap_metric"] == metric)
    for lam in sorted(lg.unique()):
        if lam > 0:
            add(f"lambda_class_gap_{lam}", lg == lam)
    add("structure_pass", df["structure_pass"].fillna(False))
    add("alignment_collapse", df["alignment_collapse"].fillna(False))
    if "proto_invalid" in df.columns:
        add("proto_invalid", df["proto_invalid"].fillna(False))
    return pd.DataFrame(summaries)


def write_reports(df: pd.DataFrame, out_dir: str) -> dict:
    out_dir = _resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_df = build_group_summaries(df)
    csv_path = os.path.join(out_dir, "round5_pretrain_diagnostics.csv")
    summary_df.to_csv(csv_path, index=False)
    md_path = os.path.join(out_dir, "round5_pretrain_diagnostics.md")
    lines = [
        "# Round 5 Pretrain Diagnostics",
        "",
        f"- Total experiments: {len(df)}",
        "",
        "## Group summaries",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "_No data_",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"csv_path": csv_path, "md_path": md_path, "summary_df": summary_df}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Round 5 pretrain diagnostics")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more optimization run dirs (pretrain/ or run root)",
    )
    parser.add_argument("--out-dir", default="result/optimization_runs/round5_combined_reports")
    args = parser.parse_args(argv)

    df = load_combined_tables(args.run_dirs)
    if df.empty:
        print("No experiments found.")
        return 1
    df = annotate_alignment_collapse(df)
    paths = write_reports(df, args.out_dir)
    print(f"Wrote {paths['csv_path']}")
    print(f"Wrote {paths['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
