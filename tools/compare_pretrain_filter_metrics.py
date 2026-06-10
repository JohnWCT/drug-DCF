#!/usr/bin/env python3
"""Compare pretrain gan_metrics against strict/relaxed visualize_vaewc_filter thresholds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualize_vaewc_results import apply_quality_filter, load_experiment_data, load_filter_config

STRICT_FILTER = {
    "enabled": True,
    "lower_is_better": ["fid", "mmd", "wasserstein", "kmeans_davies_bouldin"],
    "higher_is_better": ["kmeans_ari", "kmeans_nmi", "kmeans_silhouette", "kmeans_calinski_harabasz"],
    "thresholds": {
        "fid": 16.95,
        "mmd": 0.05,
        "wasserstein": 0.50,
        "kmeans_ari": 0.20,
        "kmeans_nmi": 0.45,
        "kmeans_silhouette": 0.15,
        "kmeans_calinski_harabasz": 370,
        "kmeans_davies_bouldin": 1.50,
    },
}


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_rows(pretrain_dir: str, label: str) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(glob(os.path.join(_resolve(pretrain_dir), "exp_*"))):
        if not os.path.isdir(exp_dir):
            continue
        row = load_experiment_data(exp_dir)
        row["backbone"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser("compare_pretrain_filter_metrics")
    parser.add_argument("--aewc-dir", default="result/benchmark_ae_vs_vae_exp746/aewc")
    parser.add_argument("--vaewc-dir", default="result/pretrain_vaewc")
    parser.add_argument("--vaewc-ids", default="exp_746", help="Comma-separated VAE exp IDs to include")
    parser.add_argument("--round3-dir", default="result/optimization_runs/vaewc_proto_infonce_round3_exp746/pretrain")
    parser.add_argument("--round3-ids", default="exp_005", help="Round3 VAE refs")
    parser.add_argument("--out", default="result/benchmark_ae_vs_vae_exp746/filter_comparison.csv")
    parser.add_argument("--relaxed-config", default="config/visualize_vaewc_filter.json")
    args = parser.parse_args()

    frames = []
    for eid in [x.strip() for x in args.vaewc_ids.split(",") if x.strip()]:
        p = os.path.join(_resolve(args.vaewc_dir), eid)
        if os.path.isdir(p):
            row = load_experiment_data(p)
            row["backbone"] = "VAE"
            row["benchmark_tag"] = "exp_746_baseline"
            frames.append(row)

    for eid in [x.strip() for x in args.round3_ids.split(",") if x.strip()]:
        p = os.path.join(_resolve(args.round3_dir), eid)
        if os.path.isdir(p):
            row = load_experiment_data(p)
            row["backbone"] = "VAE"
            row["benchmark_tag"] = "round3_best_control"
            frames.append(row)

    aewc_df = load_rows(args.aewc_dir, "AE")
    if not aewc_df.empty:
        aewc_df["benchmark_tag"] = "aewc_benchmark"
        frames.extend(aewc_df.to_dict(orient="records"))

    if not frames:
        print("No experiments found.")
        return

    df = pd.DataFrame(frames)
    relaxed_cfg = load_filter_config(_resolve(args.relaxed_config))

    strict_pass_ids = set(apply_quality_filter(df, STRICT_FILTER)["ID"].tolist())
    relaxed_pass_ids = set(apply_quality_filter(df, relaxed_cfg)["ID"].tolist())
    df["pass_strict"] = df["ID"].isin(strict_pass_ids)
    df["pass_relaxed"] = df["ID"].isin(relaxed_pass_ids)

    cols = [
        "ID", "backbone", "benchmark_tag", "lambda_proto",
        "fid", "mmd", "wasserstein",
        "kmeans_ari", "kmeans_nmi", "kmeans_silhouette", "kmeans_calinski_harabasz", "kmeans_davies_bouldin",
        "best_gan_epoch", "pass_strict", "pass_relaxed",
    ]
    cols = [c for c in cols if c in df.columns]
    out_df = df[cols].sort_values(["pass_strict", "kmeans_ari"], ascending=[False, False])

    out_path = _resolve(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_path}")
    print(f"Strict pass: {out_df['pass_strict'].sum()}/{len(out_df)}")
    print(f"Relaxed pass: {out_df['pass_relaxed'].sum()}/{len(out_df)}")


if __name__ == "__main__":
    main()
