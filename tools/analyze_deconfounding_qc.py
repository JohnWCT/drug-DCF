#!/usr/bin/env python3
"""Round 9 deconfounding quality control analyzer."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import (
    classify_deconfounding_qc,
    fit_cancer_classifier,
    fit_domain_classifier,
    iter_reproduction_models,
    latent_matrix_and_labels,
    leakage_strength,
    load_exp_metrics,
    load_latent_domain_frame,
    macro_mean,
    weighted_mean,
    write_csv,
    write_md,
)


def _conditional_summary(df: pd.DataFrame, classifiers: List[str]) -> dict:
  sufficient = df[df["sufficient_samples"] == True]  # noqa: E712
  if sufficient.empty:
    return {
      "macro_conditional_domain_auc": float("nan"),
      "weighted_conditional_domain_auc": float("nan"),
      "mean_conditional_leakage_strength": float("nan"),
      "worst_cancer_type_by_leakage": "",
    }
  primary = classifiers[0]
  auc_col = f"{primary}_domain_auc"
  leak_col = f"{primary}_leakage_strength"
  weights = sufficient["n_source"].astype(float) + sufficient["n_target"].astype(float)
  macro_auc = macro_mean(sufficient[auc_col].tolist())
  weighted_auc = weighted_mean(sufficient[auc_col].tolist(), weights.tolist())
  mean_leak = macro_mean(sufficient[leak_col].tolist())
  worst = sufficient.sort_values(leak_col, ascending=False).iloc[0]["cancer_type"]
  return {
    "macro_conditional_domain_auc": macro_auc,
    "weighted_conditional_domain_auc": weighted_auc,
    "mean_conditional_leakage_strength": mean_leak,
    "worst_cancer_type_by_leakage": worst,
  }


def analyze_model(model: dict, classifiers: List[str], min_source: int, min_target: int) -> tuple[dict, pd.DataFrame]:
    checkpoint_dir = model["checkpoint_dir"]
    metrics = load_exp_metrics(checkpoint_dir)
    try:
        frame = load_latent_domain_frame(checkpoint_dir)
        x, domain, cancer = latent_matrix_and_labels(frame)
    except Exception as exc:
        row = {
            "source_exp_id": model.get("source_exp_id", ""),
            "role": model.get("role", ""),
            "reproduction_seed": model.get("reproduction_seed", ""),
            "model_id": model.get("model_id", ""),
            "deconfounding_qc_status": "insufficient_evidence",
            "notes": str(exc),
        }
        return row, pd.DataFrame()

    global_auc, global_bal = fit_domain_classifier(x, domain, classifiers[0])
    cancer_f1, cancer_bal = fit_cancer_classifier(x, cancer)

    by_cancer_rows = []
    for cancer_type, sub in frame.groupby("cancer_type"):
        n_source = int((sub["domain"] == "source").sum())
        n_target = int((sub["domain"] == "target").sum())
        sufficient = n_source >= min_source and n_target >= min_target
        row = {
            "source_exp_id": model.get("source_exp_id", ""),
            "role": model.get("role", ""),
            "reproduction_seed": model.get("reproduction_seed", ""),
            "model_id": model.get("model_id", ""),
            "cancer_type": cancer_type,
            "n_source": n_source,
            "n_target": n_target,
            "sufficient_samples": sufficient,
        }
        if sufficient:
            sub_x, sub_domain, _ = latent_matrix_and_labels(sub)
            for clf in classifiers:
                auc, bal = fit_domain_classifier(sub_x, sub_domain, clf)
                row[f"{clf}_domain_auc"] = auc
                row[f"{clf}_domain_balanced_accuracy"] = bal
                row[f"{clf}_leakage_strength"] = leakage_strength(auc)
        by_cancer_rows.append(row)
    by_cancer_df = pd.DataFrame(by_cancer_rows)
    cond = _conditional_summary(by_cancer_df, classifiers) if not by_cancer_df.empty else {}

    status = classify_deconfounding_qc(
        global_auc,
        cond.get("macro_conditional_domain_auc", float("nan")),
        cancer_f1,
        float(metrics.get("classwise_domain_gap_inter_class_proto_margin", metrics.get("inter_class_proto_margin", np.nan))),
    )
    summary = {
        "source_exp_id": model.get("source_exp_id", ""),
        "role": model.get("role", ""),
        "reproduction_seed": model.get("reproduction_seed", ""),
        "model_id": model.get("model_id", ""),
        "global_domain_auc": global_auc,
        "global_domain_balanced_accuracy": global_bal,
        "fid": metrics.get("fid", float("nan")),
        "wasserstein": metrics.get("wasserstein", float("nan")),
        "kmeans_ari": metrics.get("kmeans_ari", float("nan")),
        "kmeans_nmi": metrics.get("kmeans_nmi", float("nan")),
        "silhouette": metrics.get("kmeans_silhouette", float("nan")),
        "davies_bouldin": metrics.get("kmeans_davies_bouldin", float("nan")),
        "cancer_classifier_macro_f1": cancer_f1,
        "cancer_classifier_balanced_accuracy": cancer_bal,
        "deconfounding_qc_status": status,
        "notes": "",
        **cond,
    }
    return summary, by_cancer_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 9 deconfounding QC")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--latent-view", default="shared")
    parser.add_argument("--min-source-per-cancer", type=int, default=10)
    parser.add_argument("--min-target-per-cancer", type=int, default=10)
    parser.add_argument("--classifiers", nargs="+", default=["logistic_regression", "small_mlp"])
    parser.add_argument("--include-fid", action="store_true")
    parser.add_argument("--include-wasserstein", action="store_true")
    parser.add_argument("--include-clustering", action="store_true")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    models = iter_reproduction_models(args.run_dir)
    summaries = []
    by_cancer = []
    for model in models:
        summary, cancer_df = analyze_model(
            model,
            args.classifiers,
            args.min_source_per_cancer,
            args.min_target_per_cancer,
        )
        summaries.append(summary)
        if not cancer_df.empty:
            by_cancer.append(cancer_df)

    outdir = args.outdir
    summary_df = pd.DataFrame(summaries)
    by_cancer_df = pd.concat(by_cancer, ignore_index=True) if by_cancer else pd.DataFrame()
    write_csv(summary_df, os.path.join(outdir, "deconfounding_qc_model_summary.csv"))
    write_csv(by_cancer_df, os.path.join(outdir, "deconfounding_qc_by_cancer.csv"))

    lines = ["# Deconfounding QC Report", "", f"Models analyzed: {len(summary_df)}", ""]
    if not summary_df.empty and "deconfounding_qc_status" in summary_df.columns:
        lines.append("## Status counts")
        for status, count in summary_df["deconfounding_qc_status"].value_counts().items():
            lines.append(f"- {status}: {count}")
    write_md(os.path.join(outdir, "deconfounding_qc_report.md"), lines)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
