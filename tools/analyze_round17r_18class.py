#!/usr/bin/env python3
"""Analyze Round 17R 18-class-clean focused rerun results."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_round17_direct_proto import (
    HISTORICAL_METRIC,
    INTEGRATED5_DRUG_METRIC,
    INTEGRATED5_METRIC,
    ROUND13_BEST,
    _feature_family,
    _normalize_aggregate,
    _seed_summary,
)
from tools.round9_diagnostics_common import load_json, resolve_path, write_csv


def _write_md(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _load_pre18_reference(round17_root: str) -> pd.DataFrame:
    candidates = [
        os.path.join(round17_root, "reports_stage17c_pre18class_fix_20260708T035033Z", "round17_top_candidates.csv"),
        os.path.join(round17_root, "reports_stage17c", "round17_top_candidates.csv"),
        os.path.join(round17_root, "reports_stage17a_pre18class_fix_20260708T035033Z", "round17_top_candidates.csv"),
    ]
    for path in candidates:
        if os.path.isfile(resolve_path(path)):
            return pd.read_csv(resolve_path(path))
    return pd.DataFrame()


def _qc_feature_root(feature_root: str) -> pd.DataFrame:
    rows = []
    root = resolve_path(feature_root)
    if not os.path.isdir(root):
        return pd.DataFrame()
    for dirpath, _, filenames in os.walk(root):
        if "feature_metadata.json" not in filenames:
            continue
        meta = load_json(os.path.join(dirpath, "feature_metadata.json"))
        rows.append(
            {
                "feature_dir": dirpath,
                "n_trainable_cancer_types": meta.get("n_trainable_cancer_types"),
                "uses_legacy_28class_cache": meta.get("uses_legacy_28class_cache"),
                "prototype_class_source": meta.get("prototype_class_source"),
                "source_prototypes_used": meta.get("source_prototypes_used"),
                "target_prototypes_used": meta.get("target_prototypes_used"),
                "feature_mode": meta.get("prototype_feature_mode") or meta.get("feature_mode"),
                "ok": (
                    int(meta.get("n_trainable_cancer_types", -1)) == 18
                    and meta.get("uses_legacy_28class_cache") is False
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_round17r(
    run_dir: str,
    settings_path: str,
    aggregate_path: Optional[str],
    stage: str,
    outdir: str,
    pre18_top_path: Optional[str] = None,
) -> dict:
    settings = load_json(settings_path)
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    agg = pd.read_csv(resolve_path(aggregate_path)) if aggregate_path and os.path.isfile(resolve_path(aggregate_path)) else pd.DataFrame()
    summary = _seed_summary(agg) if not agg.empty else pd.DataFrame()
    if not summary.empty:
        summary["feature_family"] = summary["feature_mode"].map(_feature_family)
        if "round17_model_key" in summary.columns and "model_key" not in summary.columns:
            summary["model_key"] = summary["round17_model_key"]
        summary = summary.sort_values(f"{HISTORICAL_METRIC}_mean", ascending=False)

    hist = summary.copy()
    if not hist.empty:
        hist["historical_rank"] = np.arange(1, len(hist) + 1)

    integ = summary.copy()
    if not integ.empty and f"{INTEGRATED5_METRIC}_mean" in integ.columns:
        integ = integ.sort_values(f"{INTEGRATED5_METRIC}_mean", ascending=False)
        integ["integrated5_rank"] = np.arange(1, len(integ) + 1)

    family = (
        summary.groupby("feature_family")[[f"{HISTORICAL_METRIC}_mean", f"{INTEGRATED5_METRIC}_mean"]]
        .mean()
        .reset_index()
        if not summary.empty and "feature_family" in summary.columns
        else pd.DataFrame()
    )

    round17_root = resolve_path(settings.get("round17_root", "result/optimization_runs/round17_direct_proto"))
    pre18 = pd.read_csv(resolve_path(pre18_top_path)) if pre18_top_path else _load_pre18_reference(round17_root)
    cmp_rows = []
    if not summary.empty and not pre18.empty:
        pre18 = pre18.copy()
        if "round17_model_key" in pre18.columns and "model_key" not in pre18.columns:
            pre18["model_key"] = pre18["round17_model_key"]
        for _, row in summary.iterrows():
            key = str(row.get("model_key", ""))
            mode = str(row.get("feature_mode", ""))
            match = pre18[(pre18.get("model_key", pd.Series(dtype=str)) == key) & (pre18["feature_mode"] == mode)]
            if match.empty and "model_id" in pre18.columns:
                match = pre18[pre18["model_id"].astype(str).str.contains(f"{key}_{mode}")]
            pre_mean = float(match.iloc[0][f"{HISTORICAL_METRIC}_mean"]) if not match.empty else np.nan
            cmp_rows.append(
                {
                    "model_key": key,
                    "feature_mode": mode,
                    "round17r_Average_TCGA_AUC_mean": row.get(f"{HISTORICAL_METRIC}_mean"),
                    "pre18_Average_TCGA_AUC_mean": pre_mean,
                    "delta_vs_pre18": (
                        float(row.get(f"{HISTORICAL_METRIC}_mean")) - pre_mean
                        if pd.notna(pre_mean)
                        else np.nan
                    ),
                }
            )
    comparison = pd.DataFrame(cmp_rows)

    top = summary.head(10).copy() if not summary.empty else pd.DataFrame()
    qc = _qc_feature_root(os.path.join(run_dir, "features"))

    write_csv(summary, os.path.join(outdir, "round17r_candidate_summary.csv"))
    write_csv(hist, os.path.join(outdir, "round17r_historical_ranking.csv"))
    write_csv(integ, os.path.join(outdir, "round17r_integrated5_ranking.csv"))
    write_csv(family, os.path.join(outdir, "round17r_feature_family_summary.csv"))
    write_csv(comparison, os.path.join(outdir, "round17r_vs_round17_pre18class_comparison.csv"))
    write_csv(top, os.path.join(outdir, "round17r_top_candidates.csv"))
    write_csv(qc, os.path.join(outdir, "round17r_feature_qc_summary.csv"))

    best = top.iloc[0] if not top.empty else None
    own = summary[summary["feature_mode"] == "own_plus_summary"] if not summary.empty else pd.DataFrame()
    direct = (
        summary[summary["feature_mode"].isin(["own_proto_context_projected_16", "own_proto_delta_projected_8"])]
        if not summary.empty
        else pd.DataFrame()
    )
    lines = [
        f"# Round 17R Stage {stage} Report",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "## 18-class-clean feature QC",
        f"- feature_metadata folders: **{len(qc)}**",
        f"- all ok (n=18, no legacy 28-class): **{bool(qc['ok'].all()) if not qc.empty else 'N/A'}**",
        "",
        "## Ranking snapshot (Average_TCGA_AUC)",
    ]
    if best is not None:
        lines.extend(
            [
                f"- best: `{best.get('model_id')}` / `{best.get('feature_mode')}` = "
                f"**{float(best.get(f'{HISTORICAL_METRIC}_mean')):.4f}**",
                f"- vs Round13 best ({ROUND13_BEST:.4f}): "
                f"{float(best.get(f'{HISTORICAL_METRIC}_mean')) - ROUND13_BEST:+.4f}",
            ]
        )
    if not own.empty:
        lines.append(
            f"- best own_plus_summary: **{float(own.iloc[0][f'{HISTORICAL_METRIC}_mean']):.4f}** "
            f"(`{own.iloc[0].get('model_id')}`)"
        )
    if not direct.empty:
        lines.append(
            f"- best direct prototype: **{float(direct.iloc[0][f'{HISTORICAL_METRIC}_mean']):.4f}** "
            f"(`{direct.iloc[0].get('feature_mode')}`)"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "1. 18-class-clean ranking vs pre18: see `round17r_vs_round17_pre18class_comparison.csv`",
            "2. own_plus_summary still primary? check historical + Integrated5 ranks in top candidates",
            "3. context_16 / delta_8 gap to own_plus_summary: compare top rows",
            "4. minimal_source_only_min_margin target-specific: inspect per-target columns in aggregate",
            "5. proceed to next stage only if Stage gate criteria in IDE handbook are met",
            "",
            "## Outputs",
            "- round17r_top_candidates.csv",
            "- round17r_historical_ranking.csv",
            "- round17r_integrated5_ranking.csv",
            "- round17r_feature_qc_summary.csv",
        ]
    )
    report_path = os.path.join(outdir, "round17r_final_report.md")
    _write_md(report_path, "\n".join(lines) + "\n")
    return {"outdir": outdir, "n_candidates": int(len(summary)), "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 17R results")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--settings", default="config/round17r_18class_focused_settings.json")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pre18-top", default=None)
    args = parser.parse_args()
    result = analyze_round17r(
        args.run_dir,
        args.settings,
        args.aggregate,
        args.stage,
        args.outdir,
        pre18_top_path=args.pre18_top,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
