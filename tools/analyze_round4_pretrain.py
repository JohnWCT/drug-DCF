#!/usr/bin/env python3
"""Round 4 / 4.1 pretrain diagnostics grouped by InfoNCE settings and collapse rate."""

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


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_pretrain_table(result_dir: str) -> pd.DataFrame:
    result_dir = _resolve(result_dir)
    rows = []
    for metrics_path in sorted(glob(os.path.join(result_dir, "exp_*", "gan_metrics.json"))):
        exp_id = os.path.basename(os.path.dirname(metrics_path))
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["ID"] = exp_id
        params_path = os.path.join(os.path.dirname(metrics_path), "params.json")
        if os.path.exists(params_path):
            with open(params_path, "r", encoding="utf-8") as f:
                p = json.load(f).get("params", {})
            for k in (
                "lambda_proto",
                "proto_temperature",
                "proto_direction",
                "proto_mode",
                "proto_start_epoch",
                "proto_full_epoch",
            ):
                if k not in payload and k in p:
                    payload[k] = p[k]
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return annotate_alignment_collapse(df)


def _summarize_group(g: pd.DataFrame) -> dict:
    def stat(col):
        s = pd.to_numeric(g.get(col), errors="coerce")
        return {"mean": float(s.mean()) if s.notna().any() else np.nan, "median": float(s.median()) if s.notna().any() else np.nan}

    n = len(g)
    collapse_rate = float(g["alignment_collapse"].fillna(False).mean()) if n else 0.0
    structure_rate = float(g["structure_pass"].fillna(False).mean()) if n else 0.0
    deconf_rate = float(g["deconfounding_pass"].fillna(False).mean()) if n else 0.0
    stage1_rate = float((g["structure_pass"].fillna(False) & g["deconfounding_pass"].fillna(False)).mean()) if n else 0.0
    return {
        "n": n,
        "kmeans_ari_mean": stat("kmeans_ari")["mean"],
        "kmeans_ari_median": stat("kmeans_ari")["median"],
        "fid_mean": stat("fid")["mean"],
        "fid_median": stat("fid")["median"],
        "wasserstein_mean": stat("wasserstein")["mean"],
        "wasserstein_median": stat("wasserstein")["median"],
        "collapse_rate": collapse_rate,
        "structure_pass_rate": structure_rate,
        "deconfounding_pass_rate": deconf_rate,
        "stage1_pass_rate": stage1_rate,
    }


def build_group_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summaries = []

    def add(group_name: str, mask: pd.Series):
        g = df[mask]
        row = {"group": group_name, **_summarize_group(g)}
        summaries.append(row)

    lp = pd.to_numeric(df.get("lambda_proto"), errors="coerce").fillna(0.0)
    add("lambda_proto_eq_0", lp == 0)
    add("lambda_proto_gt_0", lp > 0)
    if "proto_not_effective_checkpoint" in df.columns:
        add("proto_not_effective_true", df["proto_not_effective_checkpoint"].fillna(False))
        add("proto_not_effective_false", ~df["proto_not_effective_checkpoint"].fillna(False))
    if "proto_direction" in df.columns:
        for direction in sorted(df["proto_direction"].dropna().unique()):
            add(f"proto_direction_{direction}", df["proto_direction"] == direction)
    for lam in sorted(lp.unique()):
        add(f"lambda_proto_{lam}", lp == lam)
    if "proto_temperature" in df.columns:
        for temp in sorted(pd.to_numeric(df["proto_temperature"], errors="coerce").dropna().unique()):
            add(f"proto_temperature_{temp}", pd.to_numeric(df["proto_temperature"], errors="coerce") == temp)

    return pd.DataFrame(summaries)


def _top_examples(df: pd.DataFrame, n: int = 5):
    cols = ["ID", "lambda_proto", "kmeans_ari", "fid", "wasserstein", "alignment_collapse", "collapse_reason"]
    cols = [c for c in cols if c in df.columns]
    collapse = df[df["alignment_collapse"].fillna(False)].copy()
    collapse = collapse.sort_values(["wasserstein", "kmeans_ari"], ascending=[True, True]).head(n)
    lp = pd.to_numeric(df.get("lambda_proto"), errors="coerce").fillna(0.0)
    non_collapse = df[(lp > 0) & (~df["alignment_collapse"].fillna(False))].copy()
    non_collapse = non_collapse.sort_values(["wasserstein", "kmeans_ari"], ascending=[True, False]).head(n)
    return collapse[cols], non_collapse[cols]


def write_reports(df: pd.DataFrame, out_dir: str) -> dict:
    out_dir = _resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_df = build_group_summaries(df)
    csv_path = os.path.join(out_dir, "round4_1_pretrain_diagnostics.csv")
    summary_df.to_csv(csv_path, index=False)

    collapse_top, best_infonce = _top_examples(df)
    md_lines = [
        "# Round 4.1 Pretrain Diagnostics",
        "",
        f"- Total experiments: {len(df)}",
        "",
        "## Group summaries",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "_No data_",
        "",
        "## Top collapse examples (good wasserstein, bad structure)",
        "",
        collapse_top.to_markdown(index=False) if not collapse_top.empty else "_None_",
        "",
        "## Best non-collapse InfoNCE examples",
        "",
        best_infonce.to_markdown(index=False) if not best_infonce.empty else "_None_",
        "",
    ]
    md_path = os.path.join(out_dir, "round4_1_pretrain_diagnostics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {"csv_path": csv_path, "md_path": md_path, "summary_df": summary_df}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Round 4 pretrain collapse / InfoNCE diagnostics")
    parser.add_argument(
        "--result-dir",
        default="result/optimization_runs/vaewc_round4_cross_domain_infonce/pretrain",
        help="Directory containing exp_*/gan_metrics.json",
    )
    parser.add_argument("--out-dir", default="reports", help="Output directory for CSV/MD reports")
    args = parser.parse_args(argv)

    df = load_pretrain_table(args.result_dir)
    if df.empty:
        print(f"No experiments found under {args.result_dir}")
        return 1
    paths = write_reports(df, args.out_dir)
    print(f"Wrote {paths['csv_path']}")
    print(f"Wrote {paths['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
