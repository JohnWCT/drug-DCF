#!/usr/bin/env python3
"""Round 9 conditional domain leakage diagnostics."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import (
    fit_domain_classifier,
    iter_reproduction_models,
    latent_matrix_and_labels,
    leakage_strength,
    load_latent_domain_frame,
    macro_mean,
    weighted_mean,
    write_csv,
    write_md,
)


def analyze_model(model: dict, classifiers, min_source: int, min_target: int):
    frame = load_latent_domain_frame(model["checkpoint_dir"])
    by_cancer_rows = []
    predictions = []
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
            "notes": "",
        }
        if sufficient:
            sub_x, sub_domain, _ = latent_matrix_and_labels(sub)
            for clf in classifiers:
                auc, bal = fit_domain_classifier(sub_x, sub_domain, clf)
                row[f"{clf}_domain_auc"] = auc
                row[f"{clf}_domain_balanced_accuracy"] = bal
                row[f"{clf}_leakage_strength"] = leakage_strength(auc)
                predictions.append(
                    {
                        "model_id": model.get("model_id", ""),
                        "cancer_type": cancer_type,
                        "classifier": clf,
                        "domain_auc": auc,
                        "leakage_strength": leakage_strength(auc),
                    }
                )
        else:
            row["notes"] = "insufficient_samples"
        by_cancer_rows.append(row)
    by_cancer_df = pd.DataFrame(by_cancer_rows)
    primary = classifiers[0]
    sufficient = by_cancer_df[by_cancer_df["sufficient_samples"] == True]  # noqa: E712
    summary = {
        "source_exp_id": model.get("source_exp_id", ""),
        "role": model.get("role", ""),
        "reproduction_seed": model.get("reproduction_seed", ""),
        "model_id": model.get("model_id", ""),
        "n_cancer_types": len(by_cancer_df),
        "n_sufficient_cancer_types": len(sufficient),
        "macro_conditional_domain_auc": macro_mean(sufficient[f"{primary}_domain_auc"].tolist()) if not sufficient.empty else float("nan"),
        "mean_conditional_leakage_strength": macro_mean(sufficient[f"{primary}_leakage_strength"].tolist()) if not sufficient.empty else float("nan"),
        "weighted_conditional_domain_auc": weighted_mean(
            sufficient[f"{primary}_domain_auc"].tolist(),
            (sufficient["n_source"] + sufficient["n_target"]).astype(float).tolist(),
        )
        if not sufficient.empty
        else float("nan"),
        "worst_cancer_type_by_leakage": sufficient.sort_values(f"{primary}_leakage_strength", ascending=False).iloc[0]["cancer_type"]
        if not sufficient.empty
        else "",
    }
    return by_cancer_df, summary, pd.DataFrame(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional domain leakage diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--latent-view", default="shared")
    parser.add_argument("--min-source-per-cancer", type=int, default=10)
    parser.add_argument("--min-target-per-cancer", type=int, default=10)
    parser.add_argument("--classifiers", nargs="+", default=["logistic_regression", "small_mlp"])
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    by_cancer_all = []
    summaries = []
    predictions = []
    for model in iter_reproduction_models(args.run_dir):
        try:
            by_cancer, summary, preds = analyze_model(
                model, args.classifiers, args.min_source_per_cancer, args.min_target_per_cancer
            )
            by_cancer_all.append(by_cancer)
            summaries.append(summary)
            if not preds.empty:
                predictions.append(preds)
        except Exception as exc:
            summaries.append(
                {
                    "source_exp_id": model.get("source_exp_id", ""),
                    "model_id": model.get("model_id", ""),
                    "notes": str(exc),
                }
            )

    write_csv(pd.concat(by_cancer_all, ignore_index=True) if by_cancer_all else pd.DataFrame(), os.path.join(args.outdir, "conditional_domain_auc_by_cancer.csv"))
    write_csv(pd.DataFrame(summaries), os.path.join(args.outdir, "conditional_domain_auc_summary.csv"))
    write_csv(pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(), os.path.join(args.outdir, "domain_classifier_predictions.csv"))
    write_md(os.path.join(args.outdir, "conditional_domain_leakage_report.md"), ["# Conditional Domain Leakage", ""])
    print(f"Wrote {args.outdir}")


if __name__ == "__main__":
    main()
