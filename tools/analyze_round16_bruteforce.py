#!/usr/bin/env python3
"""Analyze Round 16 focused brute-force downstream optimization results."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round16_bruteforce_selection import aggregate_seed_stats, select_round16_bruteforce_candidates
from tools.round9_diagnostics_common import resolve_path, write_csv

ROUND13_BEST = 0.6112395039184843
ROUND15_BEST = 0.6083
STRETCH_BEST = 0.62

DISTANCE_SUMMARY_MODES = frozenset({"own_cancer", "own_plus_summary"})
OWN_CONTEXT_MODES = frozenset(
    {
        "own_proto_delta",
        "own_proto_context",
        "own_proto_context_projected_16",
        "own_proto_context_projected_32",
        "own_proto_interaction",
    }
)


def _feature_family(mode: str) -> str:
    mode = str(mode).lower()
    if mode in OWN_CONTEXT_MODES:
        return "own_context"
    if mode in DISTANCE_SUMMARY_MODES or mode.startswith("own_plus_summary"):
        return "distance_summary"
    if mode == "none":
        return "z_only"
    return "other"


def _read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.isfile(resolve_path(path)):
        return pd.DataFrame()
    return pd.read_csv(resolve_path(path))


def _metric_col(df: pd.DataFrame) -> str:
    for col in ("Average_TCGA_AUC_mean", "avg_tcga_auc_mean"):
        if col in df.columns:
            return col
    return "Average_TCGA_AUC_mean"


def _z_vs_feature_delta(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    rows = []
    metric = "mean_auc_across_seeds"
    for model_key in sorted(seed_summary["round16_model_key"].unique()):
        sub = seed_summary[seed_summary["round16_model_key"] == model_key]
        none_val = sub[sub["feature_mode"] == "none"][metric].max()
        for mode in sub["feature_mode"].unique():
            if mode == "none":
                continue
            mode_val = sub[sub["feature_mode"] == mode][metric].max()
            rows.append(
                {
                    "round16_model_key": model_key,
                    "feature_mode": mode,
                    "avg_tcga_none": none_val,
                    "avg_tcga_feature": mode_val,
                    "delta_feature_minus_none": (
                        float(mode_val - none_val) if pd.notna(none_val) and pd.notna(mode_val) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_variant_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    rows = []
    for mode, sub in seed_summary.groupby("feature_mode"):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        rows.append(
            {
                "feature_mode": mode,
                "n_combos": len(sub),
                "mean_avg_tcga": float(vals.mean()) if not vals.empty else np.nan,
                "std_avg_tcga": float(vals.std()) if len(vals) > 1 else 0.0,
                "best_avg_tcga": float(vals.max()) if not vals.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_avg_tcga", ascending=False, na_position="last")


def _model_ranking(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    rows = []
    for model_key, sub in seed_summary.groupby("round16_model_key"):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        rows.append(
            {
                "round16_model_key": model_key,
                "n_combos": len(sub),
                "mean_avg_tcga": float(vals.mean()) if not vals.empty else np.nan,
                "best_avg_tcga": float(vals.max()) if not vals.empty else np.nan,
                "best_combo_id": int(sub.loc[sub[metric].idxmax(), "combo_id"]) if not vals.empty else -1,
            }
        )
    return pd.DataFrame(rows).sort_values("best_avg_tcga", ascending=False, na_position="last")


def _feature_family_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    work = seed_summary.copy()
    work["feature_family"] = work["feature_mode"].map(_feature_family)
    rows = []
    for family, sub in work.groupby("feature_family"):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        rows.append(
            {
                "feature_family": family,
                "n_combos": len(sub),
                "mean_avg_tcga": float(vals.mean()) if not vals.empty else np.nan,
                "best_avg_tcga": float(vals.max()) if not vals.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("best_avg_tcga", ascending=False, na_position="last")


def _own_proto_context_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    sub = seed_summary[seed_summary["feature_mode"].isin(OWN_CONTEXT_MODES)]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for mode, grp in sub.groupby("feature_mode"):
        vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
        rows.append(
            {
                "feature_mode": mode,
                "n_combos": len(grp),
                "mean_avg_tcga": float(vals.mean()) if not vals.empty else np.nan,
                "best_avg_tcga": float(vals.max()) if not vals.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("best_avg_tcga", ascending=False, na_position="last")


def _context_vs_summary_delta(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    rows = []
    for model_key in sorted(seed_summary["round16_model_key"].unique()):
        sub = seed_summary[seed_summary["round16_model_key"] == model_key]
        summary_val = sub[sub["feature_mode"] == "own_plus_summary"][metric].max()
        for mode in OWN_CONTEXT_MODES:
            ctx = sub[sub["feature_mode"] == mode]
            if ctx.empty:
                continue
            ctx_val = pd.to_numeric(ctx[metric], errors="coerce").max()
            rows.append(
                {
                    "round16_model_key": model_key,
                    "context_mode": mode,
                    "avg_tcga_own_plus_summary": summary_val,
                    "avg_tcga_context_mode": ctx_val,
                    "delta_context_minus_summary": (
                        float(ctx_val - summary_val)
                        if pd.notna(summary_val) and pd.notna(ctx_val)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_report(
    *,
    stage: str,
    agg: pd.DataFrame,
    all_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
    top_candidates: pd.DataFrame,
    references: dict,
    outdir: str,
) -> str:
    r13 = float(references.get("round13_best", ROUND13_BEST))
    r15 = float(references.get("round15_best", ROUND15_BEST))
    stretch = float(references.get("stretch_best", STRETCH_BEST))

    best_row = seed_summary.sort_values("best_auc", ascending=False, na_position="last").head(1)
    best_mean_row = seed_summary.sort_values("mean_auc_across_seeds", ascending=False, na_position="last").head(1)
    best_auc = float(best_row["best_auc"].iloc[0]) if not best_row.empty else np.nan
    best_mean = float(best_mean_row["mean_auc_across_seeds"].iloc[0]) if not best_mean_row.empty else np.nan
    best_mean_combo = (
        f"{best_mean_row['round16_model_key'].iloc[0]} / {best_mean_row['feature_mode'].iloc[0]} / combo {int(best_mean_row['combo_id'].iloc[0])}"
        if not best_mean_row.empty
        else "n/a"
    )

    ten_seed_mean = np.nan
    if stage == "16b" and not seed_summary.empty:
        ten_seed_mean = float(seed_summary["mean_auc_across_seeds"].max())

    worth_16b = (
        best_mean >= r15 - 0.002
        or best_auc >= r13
        or (not seed_summary.empty and seed_summary["own_plus_delta_vs_none"].max() > 0.005)
    )
    worth_16d = best_mean >= r15 and seed_summary["std_auc_across_seeds"].min() > 0.01 if not seed_summary.empty else False

    context_summary = _own_proto_context_summary(seed_summary) if stage == "16e" else pd.DataFrame()
    context_best = float(context_summary["best_avg_tcga"].max()) if not context_summary.empty else np.nan
    summary_best = float(
        seed_summary.loc[seed_summary["feature_mode"] == "own_plus_summary", "mean_auc_across_seeds"].max()
    ) if not seed_summary.empty and (seed_summary["feature_mode"] == "own_plus_summary").any() else np.nan
    context_beats_summary = pd.notna(context_best) and pd.notna(summary_best) and context_best > summary_best

    lines = [
        "# Round 16 Brute-force Final Report",
        "",
        f"## Stage {stage.upper()} status",
        f"- Aggregate rows: {len(agg)}",
        f"- All-job rows: {len(all_df)}",
        f"- Seed-summary combos: {len(seed_summary)}",
        f"- Top candidates exported: {len(top_candidates)}",
        "",
        "## Headline metrics",
        f"- Best single-seed AUC: **{best_auc:.4f}**" if pd.notna(best_auc) else "- Best single-seed AUC: n/a",
        f"- Best seed-mean combo: **{best_mean:.4f}** ({best_mean_combo})" if pd.notna(best_mean) else "- Best seed-mean combo: n/a",
        f"- Best 10-seed mean (16B): **{ten_seed_mean:.4f}**" if pd.notna(ten_seed_mean) else "- Best 10-seed mean (16B): pending",
        "",
        "## vs references",
        f"- Round 13 peak ({r13:.4f}): {'PASS' if pd.notna(best_auc) and best_auc >= r13 else 'below' if pd.notna(best_auc) else 'pending'}",
        f"- Round 15 best ({r15:.4f}): {'PASS' if pd.notna(best_mean) and best_mean >= r15 else 'below' if pd.notna(best_mean) else 'pending'}",
        f"- Stretch target ({stretch:.4f}): {'PASS' if pd.notna(best_auc) and best_auc >= stretch else 'below' if pd.notna(best_auc) else 'pending'}",
        "",
        "## Decisions",
        f"- Run 16B confirmation: **{'YES' if worth_16b else 'NO'}**",
        f"- Run 16D micro-search: **{'YES' if worth_16d else 'NO'}**",
        "",
    ]
    if stage == "16e":
        lines.extend(
            [
                "## Round 16E own-prototype context",
                f"- Context mode beats own_plus_summary: **{'YES' if context_beats_summary else 'NO'}**",
                f"- Best context mode mean: **{context_best:.4f}**" if pd.notna(context_best) else "- Best context mode mean: n/a",
                f"- own_plus_summary best mean: **{summary_best:.4f}**" if pd.notna(summary_best) else "- own_plus_summary best mean: n/a",
                f"- Worth 10-seed confirmation: **{'YES' if worth_16b or context_beats_summary else 'NO'}**",
                "",
            ]
        )
    lines.extend(
        [
        "## Round 17 Go / No-Go",
        ]
    )
    if pd.notna(ten_seed_mean) and ten_seed_mean >= r13:
        lines.append("- **GO** final validation (10-seed mean >= Round 13 peak)")
    elif pd.notna(best_auc) and best_auc >= stretch:
        lines.append("- **GO** final validation (best >= stretch, verify seed stability)")
    elif pd.notna(best_mean) and best_mean < r15:
        lines.append("- **STOP** new optimization; validate Round 13 peak / Round 15 best / strong z-only")
    else:
        lines.append("- **CONDITIONAL** complete 16B then decide")

    report_path = os.path.join(outdir, "round16_final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def _delta_replacement_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    focus_modes = (
        "none",
        "own_plus_summary",
        "own_proto_delta_only",
        "own_plus_summary_plus_delta",
        "own_plus_summary_no_delta_control",
    )
    sub = seed_summary[seed_summary["feature_mode"].isin(focus_modes)]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (model_key, mode), grp in sub.groupby(["round16_model_key", "feature_mode"]):
        vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
        rows.append(
            {
                "round16_model_key": model_key,
                "feature_mode": mode,
                "n_combos": len(grp),
                "mean_avg_tcga": float(vals.mean()) if not vals.empty else np.nan,
                "best_avg_tcga": float(vals.max()) if not vals.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["round16_model_key", "best_avg_tcga"], ascending=[True, False])


def _delta_vs_own_plus_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    rows = []
    for model_key in sorted(seed_summary["round16_model_key"].unique()):
        sub = seed_summary[seed_summary["round16_model_key"] == model_key]
        summary_val = sub[sub["feature_mode"] == "own_plus_summary"][metric].max()
        for mode in ("own_proto_delta_only", "own_plus_summary_plus_delta"):
            mode_val = sub[sub["feature_mode"] == mode][metric].max()
            rows.append(
                {
                    "round16_model_key": model_key,
                    "compare_mode": mode,
                    "avg_tcga_own_plus_summary": summary_val,
                    "avg_tcga_compare_mode": mode_val,
                    "delta_compare_minus_summary": (
                        float(mode_val - summary_val)
                        if pd.notna(summary_val) and pd.notna(mode_val)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _delta_only_vs_none(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    rows = []
    for model_key in sorted(seed_summary["round16_model_key"].unique()):
        sub = seed_summary[seed_summary["round16_model_key"] == model_key]
        none_val = sub[sub["feature_mode"] == "none"][metric].max()
        delta_val = sub[sub["feature_mode"] == "own_proto_delta_only"][metric].max()
        rows.append(
            {
                "round16_model_key": model_key,
                "avg_tcga_none": none_val,
                "avg_tcga_delta_only": delta_val,
                "delta_only_minus_none": (
                    float(delta_val - none_val) if pd.notna(none_val) and pd.notna(delta_val) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _delta_additive_gain(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metric = "mean_auc_across_seeds"
    rows = []
    for model_key in sorted(seed_summary["round16_model_key"].unique()):
        sub = seed_summary[seed_summary["round16_model_key"] == model_key]
        summary_val = sub[sub["feature_mode"] == "own_plus_summary"][metric].max()
        plus_val = sub[sub["feature_mode"] == "own_plus_summary_plus_delta"][metric].max()
        rows.append(
            {
                "round16_model_key": model_key,
                "avg_tcga_own_plus_summary": summary_val,
                "avg_tcga_plus_delta": plus_val,
                "additive_gain": (
                    float(plus_val - summary_val) if pd.notna(summary_val) and pd.notna(plus_val) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_round16(
    run_dir: str,
    aggregate_path: str,
    stage: str = "16a",
    round13_root: Optional[str] = None,
    round15_root: Optional[str] = None,
    manifest_path: Optional[str] = None,
    outdir: Optional[str] = None,
    references: Optional[dict] = None,
) -> dict:
    del round13_root, round15_root, manifest_path
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir or os.path.join(run_dir, "reports"))
    os.makedirs(outdir, exist_ok=True)
    references = references or {}

    agg = _read_csv(aggregate_path)
    all_path = os.path.join(run_dir, "aggregate", "all_scores.csv")
    all_df = _read_csv(all_path)
    if all_df.empty:
        all_df = agg.copy()

    seed_summary = aggregate_seed_stats(all_df)
    top_candidates, _ = select_round16_bruteforce_candidates(agg, all_df, top_k=10)

    write_csv(seed_summary, os.path.join(outdir, "round16_stage16a_summary.csv"))
    write_csv(top_candidates, os.path.join(outdir, "round16_top_candidates.csv"))
    write_csv(seed_summary, os.path.join(outdir, "round16_seed_stability_summary.csv"))
    write_csv(_feature_variant_summary(seed_summary), os.path.join(outdir, "round16_feature_variant_summary.csv"))
    write_csv(seed_summary, os.path.join(outdir, "round16_model_feature_combo_summary.csv"))
    write_csv(_z_vs_feature_delta(seed_summary), os.path.join(outdir, "round16_z_vs_feature_delta.csv"))
    write_csv(_model_ranking(seed_summary), os.path.join(outdir, "round16_source_model_ranking.csv"))
    if stage == "16e":
        write_csv(_own_proto_context_summary(seed_summary), os.path.join(outdir, "round16_own_proto_context_summary.csv"))
        write_csv(_feature_family_summary(seed_summary), os.path.join(outdir, "round16_feature_family_summary.csv"))
        write_csv(_context_vs_summary_delta(seed_summary), os.path.join(outdir, "round16_context_vs_summary_delta.csv"))
    if stage == "16f":
        write_csv(_delta_replacement_summary(seed_summary), os.path.join(outdir, "round16_delta_replacement_summary.csv"))
        write_csv(_delta_vs_own_plus_summary(seed_summary), os.path.join(outdir, "round16_delta_vs_own_plus_summary.csv"))
        write_csv(_delta_only_vs_none(seed_summary), os.path.join(outdir, "round16_delta_only_vs_none.csv"))
        write_csv(_delta_additive_gain(seed_summary), os.path.join(outdir, "round16_delta_additive_gain.csv"))

    report_path = _build_report(
        stage=stage,
        agg=agg,
        all_df=all_df,
        seed_summary=seed_summary,
        top_candidates=top_candidates,
        references=references,
        outdir=outdir,
    )

    return {
        "report": report_path,
        "top_candidates": os.path.join(outdir, "round16_top_candidates.csv"),
        "seed_summary_rows": len(seed_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 16 brute-force results")
    parser.add_argument("--run-dir", default="result/optimization_runs/round16_bruteforce")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--stage", default="16a", choices=["16a", "16b", "16c", "16d", "16e", "16f"])
    parser.add_argument("--round13-root", default="result/optimization_runs/round13_proto_response")
    parser.add_argument("--round15-root", default="result/optimization_runs/round15_repro_rescue")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    aggregate = args.aggregate or os.path.join(args.run_dir, "aggregate", "aggregate_scores.csv")
    settings_path = os.path.join(PROJECT_ROOT, "config", "round16_bruteforce_settings.json")
    references = {}
    if os.path.isfile(settings_path):
        import json

        with open(settings_path, encoding="utf-8") as f:
            references = json.load(f).get("references", {})

    outputs = analyze_round16(
        run_dir=args.run_dir,
        aggregate_path=aggregate,
        stage=args.stage,
        round13_root=args.round13_root,
        round15_root=args.round15_root,
        outdir=args.outdir,
        references=references,
    )
    print(f"Wrote Round 16 report -> {outputs['report']}")


if __name__ == "__main__":
    main()
