#!/usr/bin/env python3
"""Round 6 pretrain diagnostics across tumor-topology branches."""

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

from tools.analyze_round5_pretrain import _resolve, _stat, load_combined_tables
from tools.collapse_detection import annotate_alignment_collapse
from tools.round6_selection import annotate_sweetspot_scores


ROUND6_PARAM_KEYS = (
    "lambda_tumor_topology",
    "lambda_class_gap",
    "lambda_tumor_supcon",
    "lambda_tumor_var",
    "lambda_tumor_cov",
    "use_tumor_subspace",
    "tumor_dim",
    "tumor_topology_valid",
)


def load_run_pretrain_table(result_dir: str, run_tag: str = "") -> pd.DataFrame:
    from tools.optimization_selection import enrich_selection_metadata

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
            for k in ROUND6_PARAM_KEYS + ("latent_size", "encoder_dims", "lambda_proto"):
                if k not in payload and k in p:
                    payload[k] = p[k]
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return enrich_selection_metadata(df, result_dir)


def _summarize_group(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"n": 0}
    ari = _stat(g.get("kmeans_ari"))
    wass = _stat(g.get("wasserstein"))
    fid = _stat(g.get("fid"))
    mmd = _stat(g.get("mmd"))
    sweet = _stat(g.get("sweetspot_score"))
    collapse_rate = float(g["alignment_collapse"].fillna(False).mean())
    structure_rate = float(g["structure_pass"].fillna(False).mean())
    topo_valid_rate = float(g["tumor_topology_valid"].fillna(False).mean()) if "tumor_topology_valid" in g.columns else np.nan
    sweet_pass_rate = float(g["sweetspot_pass"].fillna(False).mean()) if "sweetspot_pass" in g.columns else np.nan

    def _best_id(frame: pd.DataFrame, sort_col: str, ascending: bool) -> Optional[str]:
        if sort_col not in frame.columns or frame.empty:
            return None
        row = frame.sort_values(sort_col, ascending=ascending, na_position="last").iloc[0]
        return str(row.get("ID"))

    non_collapse = g[~g["alignment_collapse"].fillna(False)]
    return {
        "n": len(g),
        "mean_kmeans_ari": ari["mean"],
        "median_kmeans_ari": ari["median"],
        "mean_wasserstein": wass["mean"],
        "median_wasserstein": wass["median"],
        "mean_fid": fid["mean"],
        "mean_mmd": mmd["mean"],
        "mean_sweetspot_score": sweet["mean"],
        "structure_pass_rate": structure_rate,
        "collapse_rate": collapse_rate,
        "topology_valid_rate": topo_valid_rate,
        "sweetspot_pass_rate": sweet_pass_rate,
        "best_sweetspot_model": _best_id(g, "sweetspot_score", False),
        "best_kmeans_model": _best_id(g, "kmeans_ari", False),
        "best_wasserstein_model": _best_id(g, "wasserstein", True),
        "best_noncollapse_model": _best_id(non_collapse, "sweetspot_score", False),
        "best_branch_candidate": _best_id(g, "sweetspot_score", False),
    }


def build_group_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summaries = []

    def add(name: str, mask: pd.Series):
        summaries.append({"group": name, **_summarize_group(df[mask])})

    if "pretrain_run_tag" in df.columns:
        for tag in sorted(df["pretrain_run_tag"].dropna().unique()):
            add(f"branch_{tag}", df["pretrain_run_tag"] == tag)
    if "latent_size" in df.columns:
        for ls in sorted(pd.to_numeric(df["latent_size"], errors="coerce").dropna().unique()):
            add(f"latent_size_{int(ls)}", pd.to_numeric(df["latent_size"], errors="coerce") == ls)
    lt = pd.to_numeric(df.get("lambda_tumor_topology"), errors="coerce").fillna(0.0)
    for lam in sorted(lt.unique()):
        add(f"lambda_tumor_topology_{lam}", lt == lam)
    lg = pd.to_numeric(df.get("lambda_class_gap"), errors="coerce").fillna(0.0)
    for lam in sorted(lg.unique()):
        if lam > 0 or lam == 0:
            add(f"lambda_class_gap_{lam}", lg == lam)
    ls = pd.to_numeric(df.get("lambda_tumor_supcon"), errors="coerce").fillna(0.0)
    for lam in sorted(ls.unique()):
        add(f"lambda_tumor_supcon_{lam}", ls == lam)
    if "use_tumor_subspace" in df.columns:
        add("use_tumor_subspace_true", df["use_tumor_subspace"].fillna(False).astype(bool))
    add("structure_pass", df["structure_pass"].fillna(False))
    add("alignment_collapse", df["alignment_collapse"].fillna(False))
    if "sweetspot_pass" in df.columns:
        add("sweetspot_pass", df["sweetspot_pass"].fillna(False))
    if "tumor_topology_valid" in df.columns:
        add("tumor_topology_valid", df["tumor_topology_valid"].fillna(False))
    return pd.DataFrame(summaries)


def write_reports(df: pd.DataFrame, out_dir: str) -> dict:
    out_dir = _resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_df = build_group_summaries(df)
    csv_path = os.path.join(out_dir, "round6_pretrain_diagnostics.csv")
    summary_df.to_csv(csv_path, index=False)
    md_path = os.path.join(out_dir, "round6_pretrain_diagnostics.md")
    lines = [
        "# Round 6 Pretrain Diagnostics",
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
    parser = argparse.ArgumentParser(description="Round 6 pretrain diagnostics")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", default="result/optimization_runs/round6_combined/reports")
    args = parser.parse_args(argv)

    df = load_combined_tables(args.run_dirs)
    if df.empty:
        print("No experiments found.")
        return 1
    df = annotate_alignment_collapse(df)
    df = annotate_sweetspot_scores(df)
    paths = write_reports(df, args.out_dir)
    print(f"Wrote {paths['csv_path']}")
    print(f"Wrote {paths['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
