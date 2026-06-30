#!/usr/bin/env python3
"""Analyze Stage 16D pretrain filter + downstream finetune results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round16_bruteforce_selection import aggregate_seed_stats
from tools.round9_diagnostics_common import load_json, resolve_path, write_csv

ROUND13_BEST = 0.6112
ROUND15_BEST = 0.6083
STRETCH_BEST = 0.62


def _read_csv(path: str) -> pd.DataFrame:
    path = resolve_path(path)
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


def _metric_col(df: pd.DataFrame) -> str:
    for col in ("Average_TCGA_AUC_mean", "avg_tcga_auc_mean", "Global_TCGA_AUC_mean"):
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) else "value"


def _join_pretrain_downstream(candidates: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or seed_summary.empty:
        return pd.DataFrame()

    work = seed_summary.copy()
    if "downstream_model_id" not in work.columns and "model_id" in work.columns:
        work["downstream_model_id"] = work["model_id"].astype(str).str.replace(
            r"_(none|own_plus_summary)$", "", regex=True
        )
    cand = candidates.copy()
    rows = []
    for _, srow in work.iterrows():
        dm = str(srow.get("downstream_model_id", ""))
        match = cand[cand["downstream_model_id"] == dm]
        if match.empty:
            continue
        crow = match.iloc[0]
        rows.append(
            {
                "downstream_model_id": dm,
                "exp_id": crow.get("exp_id"),
                "round16_lineage": crow.get("round16_lineage"),
                "feature_mode": srow.get("feature_mode"),
                "combo_id": srow.get("combo_id"),
                "lambda_tumor_var": crow.get("lambda_tumor_var"),
                "lambda_tumor_cov": crow.get("lambda_tumor_cov"),
                "tumor_vicreg_start_epoch": crow.get("tumor_vicreg_start_epoch"),
                "vicreg_active": crow.get("vicreg_active"),
                "selection_reason": crow.get("selection_reason"),
                "kmeans_ari": crow.get("kmeans_ari"),
                "wasserstein": crow.get("wasserstein"),
                "round16d_pretrain_score": crow.get("round16d_pretrain_score"),
                "mean_auc_across_seeds": srow.get("mean_auc_across_seeds"),
                "best_auc": srow.get("best_auc"),
                "std_auc_across_seeds": srow.get("std_auc_across_seeds"),
            }
        )
    return pd.DataFrame(rows)


def _vicreg_downstream_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    group_cols = [
        "round16_lineage",
        "lambda_tumor_var",
        "lambda_tumor_cov",
        "tumor_vicreg_start_epoch",
        "vicreg_active",
    ]
    rows = []
    for keys, sub in joined.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_models"] = sub["downstream_model_id"].nunique()
        row["mean_pretrain_score"] = sub["round16d_pretrain_score"].mean()
        row["mean_kmeans_ari"] = sub["kmeans_ari"].mean()
        row["mean_wasserstein"] = sub["wasserstein"].mean()
        none_sub = sub[sub["feature_mode"] == "none"]
        sum_sub = sub[sub["feature_mode"] == "own_plus_summary"]
        row["best_downstream_none"] = (
            none_sub["mean_auc_across_seeds"].max() if not none_sub.empty else np.nan
        )
        row["best_downstream_summary"] = (
            sum_sub["mean_auc_across_seeds"].max() if not sum_sub.empty else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_stage16d(
    stage_root: str,
    *,
    candidates_path: Optional[str] = None,
    aggregate_path: Optional[str] = None,
    references: Optional[dict] = None,
    outdir: Optional[str] = None,
) -> dict:
    stage_root = resolve_path(stage_root)
    outdir = resolve_path(outdir or os.path.join(stage_root, "reports"))
    os.makedirs(outdir, exist_ok=True)
    references = references or {}

    candidates_path = resolve_path(
        candidates_path or os.path.join(outdir, "stage16d_pretrain_candidates.csv")
    )
    aggregate_path = resolve_path(
        aggregate_path or os.path.join(stage_root, "downstream", "aggregate", "aggregate_scores.csv")
    )
    candidates = _read_csv(candidates_path)
    agg = _read_csv(aggregate_path)
    all_df = _read_csv(os.path.join(stage_root, "downstream", "aggregate", "all_scores.csv"))
    if all_df.empty:
        all_df = agg.copy()

    seed_summary = aggregate_seed_stats(all_df)
    joined = _join_pretrain_downstream(candidates, seed_summary)
    vicreg_summary = _vicreg_downstream_summary(joined)

    r13 = float(references.get("round13_best", ROUND13_BEST))
    r15 = float(references.get("round15_best", ROUND15_BEST))
    stretch = float(references.get("stretch_best", STRETCH_BEST))

    best_row = seed_summary.sort_values("best_auc", ascending=False, na_position="last").head(1)
    best_mean_row = seed_summary.sort_values("mean_auc_across_seeds", ascending=False, na_position="last").head(1)
    best_auc = float(best_row["best_auc"].iloc[0]) if not best_row.empty else np.nan
    best_mean = float(best_mean_row["mean_auc_across_seeds"].iloc[0]) if not best_mean_row.empty else np.nan

    baseline_rows = joined[joined["selection_reason"] == "no_vicreg_control"]
    vicreg_rows = joined[joined["vicreg_active"] == True]  # noqa: E712
    baseline_best = float(baseline_rows["mean_auc_across_seeds"].max()) if not baseline_rows.empty else np.nan
    vicreg_best = float(vicreg_rows["mean_auc_across_seeds"].max()) if not vicreg_rows.empty else np.nan
    vicreg_helps = pd.notna(vicreg_best) and pd.notna(baseline_best) and vicreg_best > baseline_best + 0.001

    lines = [
        "# Round 16 Stage 16D — Pretrain Filter + Downstream Report",
        "",
        "## Pretrain filter",
        f"- Candidates selected: **{len(candidates)}**",
        f"- Unique checkpoints: **{candidates['pretrain_result_dir'].nunique() if not candidates.empty else 0}**",
        "",
        "## Downstream finetune",
        f"- Aggregate rows: **{len(agg)}**",
        f"- Seed-summary combos: **{len(seed_summary)}**",
        f"- Best single-seed AUC: **{best_auc:.4f}**" if pd.notna(best_auc) else "- Best single-seed AUC: n/a",
        f"- Best seed-mean AUC: **{best_mean:.4f}**" if pd.notna(best_mean) else "- Best seed-mean AUC: n/a",
        "",
        "## VICReg vs baseline (downstream)",
        f"- No-VICReg control best mean: **{baseline_best:.4f}**" if pd.notna(baseline_best) else "- No-VICReg control: n/a",
        f"- VICReg-active best mean: **{vicreg_best:.4f}**" if pd.notna(vicreg_best) else "- VICReg-active best: n/a",
        f"- VICReg improves downstream: **{'YES' if vicreg_helps else 'NO'}**",
        "",
        "## vs references",
        f"- Round 13 peak ({r13:.4f}): {'PASS' if pd.notna(best_auc) and best_auc >= r13 else 'below' if pd.notna(best_auc) else 'pending'}",
        f"- Round 15 best ({r15:.4f}): {'PASS' if pd.notna(best_mean) and best_mean >= r15 else 'below' if pd.notna(best_mean) else 'pending'}",
        f"- Stretch ({stretch:.4f}): {'PASS' if pd.notna(best_auc) and best_auc >= stretch else 'below' if pd.notna(best_auc) else 'pending'}",
        "",
        "## Top downstream combos",
    ]
    if not seed_summary.empty:
        top = seed_summary.sort_values("mean_auc_across_seeds", ascending=False).head(10)
        for i, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"{i}. `{row.get('round16_model_key', row.get('model_id', '?'))}` / "
                f"{row.get('feature_mode', '?')} — mean {row.get('mean_auc_across_seeds', np.nan):.4f}"
            )
    else:
        lines.append("- Pending downstream finetune.")

    report_path = os.path.join(outdir, "round16_stage16d_final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    paths = {
        "stage16d_pretrain_downstream_joined.csv": joined,
        "stage16d_vicreg_downstream_summary.csv": vicreg_summary,
        "stage16d_downstream_seed_summary.csv": seed_summary,
    }
    for name, df in paths.items():
        if not df.empty:
            write_csv(df, os.path.join(outdir, name))

    arch_path = _write_architecture_summary(stage_root, references, outdir)
    return {"report": report_path, "architecture_summary": arch_path, "joined_rows": len(joined)}


def _write_architecture_summary(stage_root: str, references: dict, outdir: str) -> str:
    root = resolve_path(os.path.dirname(stage_root))
    r13 = float(references.get("round13_best", ROUND13_BEST))
    r15 = float(references.get("round15_best", ROUND15_BEST))

    def _best(path: str, col: str = "Average_TCGA_AUC_mean") -> tuple[str, float]:
        df = _read_csv(path)
        if df.empty or col not in df.columns:
            return "n/a", np.nan
        idx = df[col].idxmax()
        return str(df.loc[idx].get("Model_ID", df.loc[idx].get("model_id", "?"))), float(df.loc[idx, col])

    f_best_id, f_best = _best(os.path.join(root, "stage16f/aggregate/aggregate_scores.csv"))
    e_best_id, e_best = _best(os.path.join(root, "stage16e/aggregate/aggregate_scores.csv"))
    d_best_id, d_best = _best(os.path.join(stage_root, "downstream/aggregate/aggregate_scores.csv"))

    lines = [
        "# Round 16 Architecture Round Summary (16F / 16E / 16D)",
        "",
        "| Stage | Status | Best Avg TCGA AUC | Best model |",
        "|-------|--------|-----------------|------------|",
        f"| 16F delta replacement | complete | {f_best:.4f} | `{f_best_id}` |" if pd.notna(f_best) else "| 16F | complete | n/a | n/a |",
        f"| 16E own_proto context | partial/complete | {e_best:.4f} | `{e_best_id}` |" if pd.notna(e_best) else "| 16E | in progress | n/a | n/a |",
        f"| 16D VICReg + downstream | see 16D report | {d_best:.4f} | `{d_best_id}` |" if pd.notna(d_best) else "| 16D downstream | pending | n/a | n/a |",
        "",
        f"- Reference R13 peak: **{r13:.4f}**",
        f"- Reference R15 best: **{r15:.4f}**",
        "",
        "## Takeaway",
    ]
    bests = [v for v in (f_best, e_best, d_best) if pd.notna(v)]
    if bests:
        overall = max(bests)
        lines.append(f"- Architecture-round best downstream: **{overall:.4f}**")
        lines.append(
            f"- Beats R15 ({r15:.4f}): **{'YES' if overall >= r15 else 'NO'}**; "
            f"Beats R13 ({r13:.4f}): **{'YES' if overall >= r13 else 'NO'}**"
        )
    else:
        lines.append("- Downstream results still pending.")

    arch_dir = resolve_path(os.path.join(root, "reports"))
    os.makedirs(arch_dir, exist_ok=True)
    arch_path = os.path.join(arch_dir, "round16_architecture_round_summary.md")
    with open(arch_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return arch_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 16 Stage 16D downstream")
    parser.add_argument("--stage-root", default="result/optimization_runs/round16_bruteforce/stage16d")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    settings_path = os.path.join(PROJECT_ROOT, "config", "round16_bruteforce_settings.json")
    references = load_json(settings_path).get("references", {}) if os.path.isfile(settings_path) else {}

    outputs = analyze_stage16d(
        args.stage_root,
        candidates_path=args.candidates,
        aggregate_path=args.aggregate,
        references=references,
        outdir=args.outdir,
    )
    print(f"Wrote -> {outputs['report']}")
    print(f"Architecture summary -> {outputs['architecture_summary']}")


if __name__ == "__main__":
    main()
