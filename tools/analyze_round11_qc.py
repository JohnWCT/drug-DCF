#!/usr/bin/env python3
"""Analyze Round 11 stability + SmoothL1 reconstruction ablation."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import load_json, resolve_path, write_csv, write_md

ROUND10_BEST = 0.5749
ROUND9_REPRO = 0.5671
R7_EXP048 = 0.5918
HIGH_PRIORITY_CANCERS = ["Brain", "Esophageal", "Liver", "Lung", "Ovarian"]
ROUND11A_QC_CANDIDATES = (
    "round11a_qc/reports/round11a_round10_conditional_qc.csv",
    "reports/round11a_round10_conditional_qc.csv",
)
ROUND11A_PER_CANCER_CANDIDATES = (
    "round11a_qc/reports/round11a_per_cancer_delta.csv",
    "reports/round11a_per_cancer_delta.csv",
)


def _read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.exists(resolve_path(path)):
        return pd.DataFrame()
    return pd.read_csv(resolve_path(path))


def _first_existing_csv(run_dir: str, candidates: tuple[str, ...]) -> pd.DataFrame:
    run_dir = resolve_path(run_dir)
    for rel in candidates:
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def _load_round11a_qc(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_qc = _first_existing_csv(run_dir, ROUND11A_QC_CANDIDATES)
    per_cancer = _first_existing_csv(run_dir, ROUND11A_PER_CANCER_CANDIDATES)
    return model_qc, per_cancer


def _auc_col(df: pd.DataFrame) -> Optional[str]:
    for col in (
        "logistic_regression_domain_auc",
        "small_mlp_domain_auc",
        "conditional_domain_auc",
        "domain_auc",
    ):
        if col in df.columns:
            return col
    for col in df.columns:
        if "domain_auc" in col:
            return col
    return None


def _build_per_cancer_qc_delta(
    per_cancer_df: pd.DataFrame,
    baseline_model_id: str = "exp_048",
    compare_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    if per_cancer_df.empty:
        return pd.DataFrame()

    df = per_cancer_df.copy()
    if "model_id" not in df.columns:
        return pd.DataFrame()

    cancer_col = "cancer_type" if "cancer_type" in df.columns else "cancer" if "cancer" in df.columns else None
    if cancer_col is None:
        return pd.DataFrame()

    auc_col = _auc_col(df)
    if auc_col is None:
        return pd.DataFrame()

    df[auc_col] = pd.to_numeric(df[auc_col], errors="coerce")
    baseline = df[df["model_id"].astype(str) == str(baseline_model_id)][[cancer_col, auc_col]].rename(
        columns={auc_col: "baseline_domain_auc"}
    )
    if baseline.empty:
        return pd.DataFrame()

    models = compare_models or sorted(df["model_id"].astype(str).unique().tolist())
    rows = []
    for model_id in models:
        if str(model_id) == str(baseline_model_id):
            continue
        sub = df[df["model_id"].astype(str) == str(model_id)].merge(baseline, on=cancer_col, how="left")
        sub["model_id"] = model_id
        sub["delta_domain_auc_vs_baseline"] = sub[auc_col] - sub["baseline_domain_auc"]
        sub["leakage_improved_vs_baseline"] = sub["delta_domain_auc_vs_baseline"] < 0
        sub["high_priority_cancer"] = sub[cancer_col].isin(HIGH_PRIORITY_CANCERS)
        rows.append(sub)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.rename(columns={cancer_col: "cancer_type", auc_col: "model_domain_auc"}, inplace=True)
    return out


def _collect_pretrain_summaries(run_dir: str) -> pd.DataFrame:
    pretrain_dir = os.path.join(resolve_path(run_dir), "pretrain")
    rows = []
    if not os.path.isdir(pretrain_dir):
        return pd.DataFrame()
    for exp_name in sorted(os.listdir(pretrain_dir)):
        exp_path = os.path.join(pretrain_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        row = {"model_id": exp_name, "result_dir": exp_path}
        for fname in ("run_summary.json", "gan_metrics.json"):
            fpath = os.path.join(exp_path, fname)
            if os.path.exists(fpath):
                payload = load_json(fpath)
                if fname == "run_summary.json":
                    row.update(payload.get("params", {}))
                    row.update(payload.get("metrics", {}))
                    row.update(payload.get("reconstruction_loss", {}))
                else:
                    row.update(payload)
        rows.append(row)
    return pd.DataFrame(rows)


def _exp111_leakage_summary(round11a_qc: pd.DataFrame) -> dict:
    if round11a_qc.empty or "model_id" not in round11a_qc.columns:
        return {}
    exp111 = round11a_qc[round11a_qc["model_id"].astype(str) == "exp_111"]
    exp048 = round11a_qc[round11a_qc["model_id"].astype(str) == "exp_048"]
    out = {}
    if not exp111.empty:
        row = exp111.iloc[0]
        out["exp_111_macro_conditional_domain_auc"] = row.get("macro_conditional_domain_auc")
        out["exp_111_mean_conditional_leakage_strength"] = row.get("mean_conditional_leakage_strength")
    if not exp048.empty:
        row = exp048.iloc[0]
        out["exp_048_macro_conditional_domain_auc"] = row.get("macro_conditional_domain_auc")
        out["exp_048_mean_conditional_leakage_strength"] = row.get("mean_conditional_leakage_strength")
    leak111 = out.get("exp_111_mean_conditional_leakage_strength")
    leak048 = out.get("exp_048_mean_conditional_leakage_strength")
    if pd.notna(leak111) and pd.notna(leak048):
        out["exp_111_leakage_delta_vs_exp048"] = float(leak111) - float(leak048)
        out["exp_111_leakage_improved_vs_exp048"] = float(leak111) < float(leak048)
    return out


def analyze_round11(
    run_dir: str,
    round10_root: str,
    round9_diagnostics: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
    selection_path: Optional[str] = None,
    round11a_qc_path: Optional[str] = None,
    round11a_per_cancer_path: Optional[str] = None,
) -> str:
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    summary = _collect_pretrain_summaries(run_dir)
    downstream = _read_csv(aggregate_path or os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    selection = _read_csv(selection_path or os.path.join(run_dir, "selection", "pretrain_top10.csv"))

    round11a_qc = _read_csv(round11a_qc_path) if round11a_qc_path else _first_existing_csv(run_dir, ROUND11A_QC_CANDIDATES)
    round11a_per_cancer = (
        _read_csv(round11a_per_cancer_path)
        if round11a_per_cancer_path
        else _first_existing_csv(run_dir, ROUND11A_PER_CANCER_CANDIDATES)
    )

    write_csv(summary, os.path.join(outdir, "round11_model_level_summary.csv"))
    if not round11a_qc.empty:
        write_csv(round11a_qc, os.path.join(outdir, "round11a_round10_conditional_qc_merged.csv"))

    per_cancer_delta = _build_per_cancer_qc_delta(round11a_per_cancer, baseline_model_id="exp_048")
    write_csv(per_cancer_delta, os.path.join(outdir, "round11_per_cancer_qc_delta.csv"))

    if not summary.empty and "reconstruction_loss_type" in summary.columns:
        recon = (
            summary.groupby("reconstruction_loss_type", dropna=False)
            .agg(
                n=("model_id", "count"),
                mean_wasserstein=("wasserstein", "mean"),
                mean_kmeans_ari=("kmeans_ari", "mean"),
                mean_fid=("fid", "mean"),
                mean_conditional_leakage=("mean_conditional_leakage_strength", "mean"),
            )
            .reset_index()
        )
        write_csv(recon, os.path.join(outdir, "round11_reconstruction_ablation_summary.csv"))

    if not summary.empty and "round11_branch" in summary.columns:
        cond = (
            summary.groupby("round11_branch", dropna=False)
            .agg(
                n=("model_id", "count"),
                mean_lambda_cond_adv=("lambda_cond_adv", "mean"),
                mean_wasserstein=("wasserstein", "mean"),
                mean_kmeans_ari=("kmeans_ari", "mean"),
            )
            .reset_index()
        )
        write_csv(cond, os.path.join(outdir, "round11_condadv_stability_summary.csv"))

    if not downstream.empty:
        write_csv(downstream, os.path.join(outdir, "round11_downstream_summary.csv"))

    best_avg = np.nan
    if not downstream.empty and "Average_TCGA_AUC_mean" in downstream.columns:
        best_avg = downstream["Average_TCGA_AUC_mean"].max()

    exp111_qc = _exp111_leakage_summary(round11a_qc)
    has_leakage_measurement = bool(exp111_qc) or (
        not summary.empty
        and summary.get("mean_conditional_leakage_strength", pd.Series(dtype=float)).notna().any()
    )

    round12_go = (
        pd.notna(best_avg)
        and best_avg >= ROUND10_BEST
        and has_leakage_measurement
        and exp111_qc.get("exp_111_leakage_improved_vs_exp048", False)
    )

    report_path = os.path.join(outdir, "round11_final_report.md")
    lines = [
        "# Round 11 Final Report",
        "",
        "## References",
        "",
        f"- Round 10 best (exp_111): {ROUND10_BEST}",
        f"- Round 9 reproduction: {ROUND9_REPRO}",
        f"- R7 exp_048: {R7_EXP048}",
        "",
        "## Pretrain summary",
        "",
        f"- Models with artifacts: {len(summary)}",
        f"- Selected for finetune: {len(selection)}",
    ]

    if not summary.empty and "reconstruction_loss_type" in summary.columns:
        lines.extend(["", "### Reconstruction loss coverage", ""])
        for loss_type, count in summary["reconstruction_loss_type"].value_counts().items():
            lines.append(f"- `{loss_type}`: {count} models")

    lines.extend(["", "## Round 11A conditional QC (Round 10 models)", ""])
    if round11a_qc.empty:
        lines.append("- Round 11A QC not found; run `tools/run_round11a_round10_qc.py` first.")
    else:
        lines.append(f"- Models in 11A QC table: {len(round11a_qc)}")
        if exp111_qc:
            lines.extend(
                [
                    f"- exp_111 mean conditional leakage: {exp111_qc.get('exp_111_mean_conditional_leakage_strength', 'NA')}",
                    f"- exp_048 mean conditional leakage: {exp111_qc.get('exp_048_mean_conditional_leakage_strength', 'NA')}",
                    f"- exp_111 leakage delta vs exp_048: {exp111_qc.get('exp_111_leakage_delta_vs_exp048', 'NA')}",
                    f"- exp_111 improved vs exp_048: {exp111_qc.get('exp_111_leakage_improved_vs_exp048', 'NA')}",
                ]
            )

    lines.extend(["", "## Per-cancer conditional leakage delta", ""])
    if per_cancer_delta.empty:
        lines.append("- No per-cancer delta available (missing Round 11A per-cancer QC).")
    else:
        hp = per_cancer_delta[per_cancer_delta.get("high_priority_cancer", False) == True]  # noqa: E712
        if not hp.empty and "model_id" in hp.columns:
            for model_id in sorted(hp["model_id"].astype(str).unique()):
                sub = hp[hp["model_id"].astype(str) == model_id]
                improved = int(sub["leakage_improved_vs_baseline"].sum())
                lines.append(
                    f"- `{model_id}` high-priority cancers improved vs exp_048: {improved}/{len(sub)}"
                )
        lines.append(f"- Full table: `round11_per_cancer_qc_delta.csv` ({len(per_cancer_delta)} rows)")

    lines.extend(
        [
            "",
            "## Downstream",
            "",
            f"- Best Average_TCGA_AUC_mean: {best_avg}",
            "",
            "## Round 12 decision",
            "",
            f"**Recommendation:** `{'go_prototype_alignment' if round12_go else 'defer_round12'}`",
            "",
            "Round 12 requires measured conditional leakage improvement (11A), cancer retention,",
            "and downstream >= Round 10 exp_111.",
        ]
    )
    write_md(report_path, lines)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 11 QC")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round10-root", default="result/optimization_runs/round10_cond_adv")
    parser.add_argument("--round9-diagnostics", default="result/optimization_runs/round9_diagnostics/final_report")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--round11a-qc", default=None, help="Optional path to round11a_round10_conditional_qc.csv")
    parser.add_argument("--round11a-per-cancer", default=None, help="Optional path to round11a_per_cancer_delta.csv")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(args.run_dir, "final_report")
    path = analyze_round11(
        run_dir=args.run_dir,
        round10_root=args.round10_root,
        round9_diagnostics=args.round9_diagnostics,
        outdir=outdir,
        aggregate_path=args.aggregate,
        selection_path=args.selection,
        round11a_qc_path=args.round11a_qc,
        round11a_per_cancer_path=args.round11a_per_cancer,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
