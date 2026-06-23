#!/usr/bin/env python3
"""Round 12A: prototype-gap diagnostics on Round 11 top candidates."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_cancer_prototypes import analyze_model
from tools.round9_diagnostics_common import resolve_path, write_csv, write_md

SUMMARY_COLUMNS = [
    "model_id",
    "source_exp_id",
    "round11_branch",
    "Average_TCGA_AUC_mean",
    "Global_TCGA_AUC_mean",
    "reconstruction_loss_type",
    "global_adv_mode",
    "lambda_cond_adv",
    "lambda_global_adv_multiplier",
    "mean_same_cancer_proto_distance",
    "weighted_same_cancer_proto_distance",
    "worst_cancer_by_proto_gap",
    "inter_cancer_margin",
    "prototype_alignment_ratio",
    "macro_conditional_domain_auc",
    "mean_conditional_leakage_strength",
    "kmeans_ari",
    "wasserstein",
    "fid",
]


def _model_id_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("ID", "model_id", "exp_id"):
        if col in df.columns:
            return col
    return None


def _collect_model_ids(
    round11_root: str,
    selection_path: str,
    top_k: int,
    force_models: List[str],
) -> List[str]:
    ids: Set[str] = set(str(m) for m in force_models)
    sel_path = resolve_path(selection_path)
    if os.path.isfile(sel_path):
        sel = pd.read_csv(sel_path)
        id_col = _model_id_col(sel)
        if id_col:
            ids.update(sel[id_col].astype(str).head(top_k).tolist())
    agg_path = os.path.join(resolve_path(round11_root), "aggregate", "aggregate_scores.csv")
    if os.path.isfile(agg_path):
        agg = pd.read_csv(agg_path)
        id_col = _model_id_col(agg)
        if id_col and "Average_TCGA_AUC_mean" in agg.columns:
            top = agg.sort_values("Average_TCGA_AUC_mean", ascending=False).head(top_k)
            ids.update(top[id_col].astype(str).tolist())
    return sorted(ids)


def _load_run_summary_metrics(checkpoint_dir: str) -> dict:
    import json

    summary_path = os.path.join(checkpoint_dir, "run_summary.json")
    if not os.path.isfile(summary_path):
        return {}
    with open(summary_path, encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    params = payload.get("params", {})
    out = dict(metrics)
    out.update({k: params.get(k) for k in params if k not in out})
    return out


def analyze_round12_baseline_gaps(
    round11_root: str,
    outdir: str,
    selection_path: str,
    top_k: int = 30,
    force_models: Optional[List[str]] = None,
) -> str:
    round11_root = resolve_path(round11_root)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    force_models = force_models or ["exp_035", "exp_111"]

    agg_path = os.path.join(round11_root, "aggregate", "aggregate_scores.csv")
    aggregate = pd.read_csv(agg_path) if os.path.isfile(agg_path) else pd.DataFrame()
    id_col = _model_id_col(aggregate) if not aggregate.empty else "model_id"

    model_ids = _collect_model_ids(
        round11_root,
        selection_path,
        top_k=top_k,
        force_models=force_models,
    )

    per_cancer_rows = []
    summary_rows = []

    for model_id in model_ids:
        checkpoint_dir = os.path.join(round11_root, "pretrain", model_id)
        if not os.path.isdir(checkpoint_dir):
            continue
        model = {
            "model_id": model_id,
            "checkpoint_dir": checkpoint_dir,
            "source_exp_id": model_id,
        }
        try:
            by_cancer, summary, _, _ = analyze_model(
                model, metrics=["cosine", "euclidean"], min_source=2, min_target=2
            )
            by_cancer["model_id"] = model_id
            per_cancer_rows.append(by_cancer)
        except Exception as exc:
            summary = {"model_id": model_id, "notes": str(exc)}

        run_metrics = _load_run_summary_metrics(checkpoint_dir)
        agg_row = {}
        if not aggregate.empty and id_col:
            match = aggregate[aggregate[id_col].astype(str) == str(model_id)]
            if not match.empty:
                agg_row = match.iloc[0].to_dict()

        row = {
            "model_id": model_id,
            "source_exp_id": model_id,
            "round11_branch": run_metrics.get("round11_branch", agg_row.get("round11_branch", "")),
            "Average_TCGA_AUC_mean": agg_row.get("Average_TCGA_AUC_mean", np.nan),
            "Global_TCGA_AUC_mean": agg_row.get("Global_TCGA_AUC_mean", np.nan),
            "reconstruction_loss_type": run_metrics.get(
                "reconstruction_loss_type", agg_row.get("reconstruction_loss_type", "mse")
            ),
            "global_adv_mode": run_metrics.get("global_adv_mode", ""),
            "lambda_cond_adv": run_metrics.get("lambda_cond_adv", np.nan),
            "lambda_global_adv_multiplier": run_metrics.get(
                "lambda_global_adv_multiplier", np.nan
            ),
            "mean_same_cancer_proto_distance": summary.get(
                "mean_same_cancer_source_target_cosine_distance", np.nan
            ),
            "weighted_same_cancer_proto_distance": summary.get(
                "mean_same_cancer_source_target_euclidean_distance", np.nan
            ),
            "worst_cancer_by_proto_gap": summary.get("worst_aligned_cancer_type", ""),
            "inter_cancer_margin": summary.get("mean_inter_cancer_target_margin", np.nan),
            "prototype_alignment_ratio": summary.get("prototype_alignment_ratio", np.nan),
            "macro_conditional_domain_auc": run_metrics.get(
                "macro_conditional_domain_auc", np.nan
            ),
            "mean_conditional_leakage_strength": run_metrics.get(
                "mean_conditional_leakage_strength", np.nan
            ),
            "kmeans_ari": run_metrics.get("kmeans_ari", np.nan),
            "wasserstein": run_metrics.get("wasserstein", np.nan),
            "fid": run_metrics.get("fid", np.nan),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    per_cancer_df = (
        pd.concat(per_cancer_rows, ignore_index=True) if per_cancer_rows else pd.DataFrame()
    )

    write_csv(summary_df, os.path.join(outdir, "round11_top_prototype_gap_summary.csv"))
    write_csv(per_cancer_df, os.path.join(outdir, "round11_per_cancer_prototype_gap.csv"))

    best_proto = summary_df.sort_values(
        "mean_same_cancer_proto_distance", ascending=True, na_position="last"
    ).head(1)
    best_down = summary_df.sort_values(
        "Average_TCGA_AUC_mean", ascending=False, na_position="last"
    ).head(1)

    lines = [
        "# Round 12A Baseline Prototype Gap QC",
        "",
        f"- Models analyzed: {len(summary_df)}",
        f"- Round 11 root: `{round11_root}`",
        "",
    ]
    if not best_down.empty:
        r = best_down.iloc[0]
        lines.append(
            f"- Best downstream: **{r['model_id']}** "
            f"(Avg TCGA={r.get('Average_TCGA_AUC_mean', 'n/a')})"
        )
    if not best_proto.empty:
        r = best_proto.iloc[0]
        lines.append(
            f"- Lowest prototype gap: **{r['model_id']}** "
            f"(mean same-cancer distance={r.get('mean_same_cancer_proto_distance', 'n/a')})"
        )
    lines.extend(
        [
            "",
            "Primary baseline for Round 12 sweeps: **exp_035** (11B 10C stabilization).",
        ]
    )
    report_path = os.path.join(outdir, "round12a_baseline_qc_report.md")
    write_md(report_path, lines)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 12A Round 11 baseline prototype gaps")
    parser.add_argument(
        "--round11-root",
        default="result/optimization_runs/round11_stability_recon",
    )
    parser.add_argument(
        "--selection",
        default="result/optimization_runs/round11_stability_recon/selection/pretrain_top10.csv",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--force-models", nargs="*", default=["exp_035", "exp_111"])
    args = parser.parse_args()

    report = analyze_round12_baseline_gaps(
        round11_root=args.round11_root,
        outdir=args.outdir,
        selection_path=args.selection,
        top_k=args.top_k,
        force_models=args.force_models,
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
