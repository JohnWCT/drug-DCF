#!/usr/bin/env python3
"""Integrate Round 9 diagnostics outputs into final reports."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import find_existing_tsne_path, resolve_path, write_csv, write_md


def _read_csv(path: str) -> pd.DataFrame:
    path = resolve_path(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_spearman(x, y) -> tuple[float, float, int]:
    mask = ~(pd.isna(x) | pd.isna(y))
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    corr, pval = spearmanr(x[mask], y[mask])
    return float(corr), float(pval), n


def build_model_level_summary(
    aggregate_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    proto_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    resolved_df: pd.DataFrame,
) -> pd.DataFrame:
    agg = aggregate_df.copy()
    if agg.empty:
        base = qc_df.copy()
    else:
        base = agg.rename(columns={"Model_ID": "model_id"}) if "Model_ID" in agg.columns else agg.copy()
    for extra, key in [
        (qc_df, "model_id"),
        (cond_df, "model_id"),
        (proto_df, "model_id"),
        (stability_df, "model_id"),
    ]:
        if extra.empty or key not in extra.columns:
            continue
        drop_cols = [c for c in extra.columns if c in base.columns and c != key]
        base = base.merge(extra.drop(columns=drop_cols, errors="ignore"), on=key, how="left")
    if "checkpoint_dir" not in base.columns and not resolved_df.empty:
        pass
    if "result_folder" in base.columns:
        base["existing_tsne_path"] = base["result_folder"].map(
            lambda p: find_existing_tsne_path(p) if isinstance(p, str) and p else ""
        )
    return base


def build_seed_reproducibility(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty or "source_exp_id" not in model_df.columns:
        return pd.DataFrame()
    rows = []
    metric_cols = [
        c
        for c in [
            "Average_TCGA_AUC_mean",
            "Global_TCGA_AUC_mean",
            "fid",
            "wasserstein",
            "macro_conditional_domain_auc",
            "mean_conditional_leakage_strength",
            "mean_same_cancer_source_target_cosine_distance",
            "active_dim_count_std_gt_0_01",
        ]
        if c in model_df.columns
    ]
    for source_exp, sub in model_df.groupby("source_exp_id"):
        row = {
            "source_exp_id": source_exp,
            "role": sub["role"].iloc[0] if "role" in sub.columns else "",
            "n_seeds": len(sub),
        }
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())
        avg_std = row.get("Average_TCGA_AUC_mean_std", float("nan"))
        row["reproducibility_flag"] = "stable" if avg_std <= 0.01 else "variable"
        rows.append(row)
    return pd.DataFrame(rows)


def build_downstream_correlation(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty or "Average_TCGA_AUC_mean" not in model_df.columns:
        return pd.DataFrame()
    target = pd.to_numeric(model_df["Average_TCGA_AUC_mean"], errors="coerce")
    pairs = [
        ("fid", "FID"),
        ("wasserstein", "Wasserstein"),
        ("kmeans_ari", "kmeans_ari"),
        ("global_domain_auc", "global_domain_auc"),
        ("macro_conditional_domain_auc", "macro_conditional_domain_auc"),
        ("mean_conditional_leakage_strength", "conditional_leakage_strength"),
        ("mean_same_cancer_source_target_cosine_distance", "same_cancer_prototype_distance"),
        ("mean_inter_cancer_source_margin", "inter_cancer_margin"),
        ("active_dim_count_std_gt_0_01", "active_dim_count"),
        ("effective_rank", "effective_rank"),
    ]
    rows = []
    for col, label in pairs:
        if col not in model_df.columns:
            continue
        corr, pval, n = _safe_spearman(pd.to_numeric(model_df[col], errors="coerce"), target)
        rows.append(
            {
                "metric_x": label,
                "metric_y": "Average_TCGA_AUC_mean",
                "spearman_corr": corr,
                "p_value": pval,
                "n": n,
                "low_n_warning": n < 6,
            }
        )
    return pd.DataFrame(rows)


def build_per_cancer_problem_list(by_cancer_df: pd.DataFrame) -> pd.DataFrame:
    if by_cancer_df.empty:
        return pd.DataFrame()
    primary_auc = "logistic_domain_auc" if "logistic_domain_auc" in by_cancer_df.columns else "logistic_regression_domain_auc"
    leak_col = "logistic_leakage_strength" if "logistic_leakage_strength" in by_cancer_df.columns else f"logistic_regression_leakage_strength"
    if primary_auc not in by_cancer_df.columns:
        return pd.DataFrame()
    grouped = by_cancer_df.groupby("cancer_type").agg(
        n_source=("n_source", "mean"),
        n_target=("n_target", "mean"),
        sufficient_samples=("sufficient_samples", "max"),
        mean_conditional_domain_auc=(primary_auc, "mean"),
        conditional_leakage_strength=(leak_col, "mean"),
        mean_source_target_proto_distance=("source_target_cosine_distance", "mean"),
    ).reset_index()
    grouped["rank_by_leakage"] = grouped["conditional_leakage_strength"].rank(ascending=False, method="dense")
    grouped["rank_by_proto_distance"] = grouped["mean_source_target_proto_distance"].rank(ascending=False, method="dense")
    grouped["inter_cancer_margin"] = float("nan")
    grouped["insufficient_samples"] = ~grouped["sufficient_samples"].astype(bool)

    def priority(row):
        if row["insufficient_samples"]:
            return "insufficient"
        leak_high = row["conditional_leakage_strength"] >= grouped["conditional_leakage_strength"].median()
        proto_high = row["mean_source_target_proto_distance"] >= grouped["mean_source_target_proto_distance"].median()
        if leak_high and proto_high:
            return "high"
        if leak_high or proto_high:
            return "medium"
        return "low"

    grouped["recommended_round10_priority"] = grouped.apply(priority, axis=1)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 9 final diagnostics report")
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--resolved-baselines", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    qc_df = _read_csv(os.path.join(args.diagnostics_dir, "deconfounding_qc_model_summary.csv"))
    cond_summary = _read_csv(os.path.join(args.diagnostics_dir, "conditional_domain_auc_summary.csv"))
    cond_by_cancer = _read_csv(os.path.join(args.diagnostics_dir, "conditional_domain_auc_by_cancer.csv"))
    proto_summary = _read_csv(os.path.join(args.diagnostics_dir, "prototype_margin_summary.csv"))
    proto_by_cancer = _read_csv(os.path.join(args.diagnostics_dir, "prototype_distance_by_cancer.csv"))
    stability_df = _read_csv(os.path.join(args.diagnostics_dir, "latent_stability_by_model.csv"))
    aggregate_df = _read_csv(args.aggregate)
    resolved_df = _read_csv(args.resolved_baselines)

    if not proto_by_cancer.empty and not cond_by_cancer.empty:
        cond_by_cancer = cond_by_cancer.merge(
            proto_by_cancer[
                ["model_id", "cancer_type", "source_target_cosine_distance"]
            ],
            on=["model_id", "cancer_type"],
            how="left",
        )

    model_df = build_model_level_summary(aggregate_df, qc_df, cond_summary, proto_summary, stability_df, resolved_df)
    seed_df = build_seed_reproducibility(model_df)
    corr_df = build_downstream_correlation(model_df)
    cancer_df = build_per_cancer_problem_list(cond_by_cancer)

    write_csv(model_df, os.path.join(outdir, "round9_model_level_summary.csv"))
    write_csv(seed_df, os.path.join(outdir, "round9_seed_reproducibility_summary.csv"))
    write_csv(qc_df, os.path.join(outdir, "round9_deconfounding_qc_summary.csv"))
    write_csv(corr_df, os.path.join(outdir, "round9_diagnostics_downstream_correlation.csv"))
    write_csv(cancer_df, os.path.join(outdir, "round9_per_cancer_problem_list.csv"))

    lines = [
        "# Round 9 Final Report",
        "",
        f"- Models in summary: **{len(model_df)}**",
        f"- Baselines resolved: **{len(resolved_df)}**",
        "",
        "## Deconfounding QC status",
    ]
    if not qc_df.empty and "deconfounding_qc_status" in qc_df.columns:
        for status, count in qc_df["deconfounding_qc_status"].value_counts().items():
            lines.append(f"- {status}: {count}")
    if not corr_df.empty:
        lines.extend(["", "## Diagnostics ↔ downstream (exploratory)"])
        for _, row in corr_df.iterrows():
            warn = " (low n)" if row.get("low_n_warning") else ""
            lines.append(
                f"- {row['metric_x']} vs Avg TCGA: spearman={row['spearman_corr']:.3f}, n={int(row['n'])}{warn}"
            )
    write_md(os.path.join(outdir, "round9_final_report.md"), lines)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
