#!/usr/bin/env python3
"""Round 11A: post-hoc conditional QC on Round 10 selected models."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Set

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_conditional_domain_leakage import analyze_model
from tools.round9_diagnostics_common import load_json, resolve_path, write_csv, write_md

ROUND10_BEST_AVG_TCGA = 0.5749
ROUND9_REPRO_BEST = 0.5671


def _model_checkpoint(round10_root: str, model_id: str) -> str:
    return os.path.join(resolve_path(round10_root), "pretrain", model_id)


def _collect_qc_model_ids(
    round10_root: str,
    force_models: List[str],
    top_k: int,
) -> List[str]:
    round10_root = resolve_path(round10_root)
    selected: Set[str] = set(force_models)

    selection_path = os.path.join(round10_root, "selection", "pretrain_top10.csv")
    if os.path.exists(selection_path):
        sel = pd.read_csv(selection_path)
        id_col = "ID" if "ID" in sel.columns else "model_id" if "model_id" in sel.columns else None
        if id_col:
            for mid in sel[id_col].astype(str).head(top_k):
                selected.add(mid)

    aggregate_path = os.path.join(round10_root, "aggregate", "aggregate_scores.csv")
    if os.path.exists(aggregate_path):
        agg = pd.read_csv(aggregate_path)
        if "Average_TCGA_AUC_mean" in agg.columns:
            for mid in agg.nlargest(5, "Average_TCGA_AUC_mean")["Model_ID"].astype(str):
                selected.add(mid)
        if "Global_TCGA_AUC_mean" in agg.columns:
            for mid in agg.nlargest(5, "Global_TCGA_AUC_mean")["Model_ID"].astype(str):
                selected.add(mid)

    manifest_path = os.path.join(round10_root, "manifests", "pretrain_sweep_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        mask_10a = manifest.get("round10_branch", "").astype(str).str.contains("10A", na=False)
        success = manifest.get("status", "") == "success"
        ten_a = manifest[mask_10a & success]
        if not ten_a.empty:
            best = ten_a.sort_values("lambda_cond_adv", na_position="last").iloc[0]
            exp_id = os.path.basename(str(best["result_dir"]).rstrip("/"))
            selected.add(exp_id)

    for mid in ["exp_048", "exp_111"]:
        if os.path.isdir(_model_checkpoint(round10_root, mid)):
            selected.add(mid)

    return sorted(selected)


def run_round11a_qc(
    round10_root: str,
    outdir: str,
    force_models: List[str] | None = None,
    top_k: int = 24,
    classifiers: List[str] | None = None,
) -> str:
    round10_root = resolve_path(round10_root)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    reports_dir = os.path.join(outdir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    force_models = force_models or ["exp_111"]
    classifiers = classifiers or ["logistic_regression", "small_mlp"]
    model_ids = _collect_qc_model_ids(round10_root, force_models, top_k)

    by_cancer_all = []
    summaries = []
    for model_id in model_ids:
        ckpt = _model_checkpoint(round10_root, model_id)
        if not os.path.isdir(ckpt):
            summaries.append({"model_id": model_id, "notes": "checkpoint_missing"})
            continue
        model = {
            "model_id": model_id,
            "checkpoint_dir": ckpt,
            "source_exp_id": "exp_048",
            "role": "round10_pretrain",
            "reproduction_seed": "",
        }
        try:
            by_cancer, summary, _preds = analyze_model(model, classifiers, 10, 10)
            by_cancer["model_id"] = model_id
            by_cancer_all.append(by_cancer)
            summaries.append(summary)
        except Exception as exc:
            summaries.append({"model_id": model_id, "notes": str(exc)})

    summary_df = pd.DataFrame(summaries)
    by_cancer_df = pd.concat(by_cancer_all, ignore_index=True) if by_cancer_all else pd.DataFrame()

    qc_path = os.path.join(reports_dir, "round11a_round10_conditional_qc.csv")
    write_csv(summary_df, qc_path)

    delta_path = os.path.join(reports_dir, "round11a_per_cancer_delta.csv")
    write_csv(by_cancer_df, delta_path)

    exp111_row = summary_df[summary_df["model_id"] == "exp_111"]
    exp111_md = os.path.join(reports_dir, "round11a_exp111_qc_report.md")
    lines = [
        "# Round 11A exp_111 QC Report",
        "",
        f"- Models analyzed: {len(model_ids)}",
        f"- Round 10 best Avg TCGA reference: {ROUND10_BEST_AVG_TCGA}",
        f"- Round 9 reproduction best: {ROUND9_REPRO_BEST}",
        "",
    ]
    if not exp111_row.empty:
        row = exp111_row.iloc[0]
        lines.extend(
            [
                "## exp_111 metrics",
                "",
                f"- macro_conditional_domain_auc: {row.get('macro_conditional_domain_auc', 'NA')}",
                f"- mean_conditional_leakage_strength: {row.get('mean_conditional_leakage_strength', 'NA')}",
                f"- n_sufficient_cancer_types: {row.get('n_sufficient_cancer_types', 'NA')}",
                "",
            ]
        )
    write_md(exp111_md, lines)

    go_no_go = "defer_round11b"
    if not exp111_row.empty:
        leak = exp111_row.iloc[0].get("mean_conditional_leakage_strength")
        if pd.notna(leak) and float(leak) < 0.35:
            go_no_go = "proceed_10c_stabilization"

    go_path = os.path.join(reports_dir, "round11a_go_no_go.md")
    write_md(
        go_path,
        [
            "# Round 11A Go / No-Go",
            "",
            f"**Recommendation:** `{go_no_go}`",
            "",
            "Proceed with Round 11B/11C if exp_111 shows lower conditional leakage than Round 9 exp_048",
            "while maintaining kmeans_ari and downstream > Round 9 reproduction best.",
            "",
            f"See `{qc_path}` for full model-level QC.",
        ],
    )
    return qc_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 11A Round 10 post-hoc conditional QC")
    parser.add_argument("--round10-root", default="result/optimization_runs/round10_cond_adv")
    parser.add_argument("--round9-diagnostics", default="result/optimization_runs/round9_diagnostics/final_report")
    parser.add_argument("--outdir", default="result/optimization_runs/round11_stability_recon/round11a_qc")
    parser.add_argument("--force-models", default="exp_111")
    parser.add_argument("--top-k", type=int, default=24)
    args = parser.parse_args()

    force = [m.strip() for m in args.force_models.split(",") if m.strip()]
    path = run_round11a_qc(
        round10_root=args.round10_root,
        outdir=args.outdir,
        force_models=force,
        top_k=args.top_k,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
