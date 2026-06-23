#!/usr/bin/env python3
"""Analyze Round 13 prototype-distance response feature ablation results."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import resolve_path, write_csv, write_md

ROUND12_BEST = 0.5971789386885913
ROUND11_BEST = 0.5828
R7_BEST = 0.5918
STRONG_SUCCESS = 0.6000


def _best_row(df: pd.DataFrame, col: str = "Average_TCGA_AUC_mean") -> Optional[pd.Series]:
    if df.empty or col not in df.columns:
        return None
    work = df.copy()
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[col])
    if work.empty:
        return None
    id_col = "Model_ID" if "Model_ID" in work.columns else "ID"
    return work.sort_values(col, ascending=False).iloc[0]


def _infer_feature_mode(model_id: str) -> str:
    text = str(model_id)
    for suffix in (
        "own_plus_summary",
        "all_source_and_target",
        "all_source_anchors",
        "own_cancer",
        "none",
    ):
        if text.endswith(suffix):
            return suffix
    return "none"


def _infer_source_model_id(model_id: str) -> str:
    text = str(model_id)
    if text.startswith("r13_"):
        body = text[len("r13_") :]
        for suffix in (
            "_own_plus_summary",
            "_all_source_and_target",
            "_all_source_anchors",
            "_own_cancer",
            "_none",
        ):
            if body.endswith(suffix):
                return body[: -len(suffix)]
    return text


def _z_vs_proto_delta(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    id_col = "Model_ID" if "Model_ID" in aggregate.columns else "ID"
    work = aggregate.copy()
    if "source_model_id" not in work.columns:
        work["source_model_id"] = work[id_col].map(_infer_source_model_id)
    if "prototype_feature_mode" not in work.columns:
        work["prototype_feature_mode"] = work[id_col].map(_infer_feature_mode)
    rows = []
    for source_id, sub in work.groupby("source_model_id"):
        z_only = sub[sub["prototype_feature_mode"].astype(str) == "none"]
        proto = sub[sub["prototype_feature_mode"].astype(str) != "none"]
        if z_only.empty or proto.empty:
            continue
        z_best = _best_row(z_only)
        p_best = _best_row(proto)
        if z_best is None or p_best is None:
            continue
        rows.append(
            {
                "source_model_id": source_id,
                "z_only_model_id": z_best.get(id_col),
                "z_only_avg_tcga": z_best.get("Average_TCGA_AUC_mean"),
                "best_proto_model_id": p_best.get(id_col),
                "best_proto_feature_mode": p_best.get("prototype_feature_mode", p_best.get("feature_mode")),
                "best_proto_avg_tcga": p_best.get("Average_TCGA_AUC_mean"),
                "delta_proto_minus_z_only": float(p_best.get("Average_TCGA_AUC_mean", np.nan))
                - float(z_best.get("Average_TCGA_AUC_mean", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def analyze_round13(
    run_dir: str,
    round12_root: str,
    round11_root: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
) -> str:
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    agg_path = resolve_path(aggregate_path or os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    aggregate = pd.read_csv(agg_path) if os.path.isfile(agg_path) else pd.DataFrame()

    manifest_path = os.path.join(run_dir, "manifests", "finetune_dispatch_manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.isfile(manifest_path) else pd.DataFrame()
    proto_manifest_path = os.path.join(run_dir, "manifests", "proto_feature_manifest.csv")
    proto_manifest = pd.read_csv(proto_manifest_path) if os.path.isfile(proto_manifest_path) else pd.DataFrame()

    if not aggregate.empty and "prototype_feature_mode" not in aggregate.columns and not manifest.empty:
        id_col = "Model_ID" if "Model_ID" in aggregate.columns else "ID"
        meta = manifest.drop_duplicates(subset=["model_id"])[
            ["model_id", "prototype_feature_mode", "response_input_mode", "source_model_id"]
        ]
        aggregate = aggregate.merge(meta, left_on=id_col, right_on="model_id", how="left")

    feature_summary = []
    if not aggregate.empty and "prototype_feature_mode" in aggregate.columns:
        for mode, sub in aggregate.groupby("prototype_feature_mode"):
            best = _best_row(sub)
            feature_summary.append(
                {
                    "prototype_feature_mode": mode,
                    "n_models": len(sub),
                    "best_model_id": best.get("Model_ID", best.get("ID")) if best is not None else "",
                    "best_avg_tcga": best.get("Average_TCGA_AUC_mean") if best is not None else np.nan,
                    "mean_avg_tcga": pd.to_numeric(sub["Average_TCGA_AUC_mean"], errors="coerce").mean(),
                }
            )
    feature_mode_df = pd.DataFrame(feature_summary)
    write_csv(feature_mode_df, os.path.join(outdir, "round13_feature_mode_summary.csv"))
    write_csv(aggregate, os.path.join(outdir, "round13_model_level_summary.csv"))

    delta_df = _z_vs_proto_delta(aggregate)
    write_csv(delta_df, os.path.join(outdir, "round13_z_vs_proto_delta.csv"))

    dim_rows = []
    feat_root = os.path.join(run_dir, "features")
    for meta_path in glob.glob(os.path.join(feat_root, "*", "*", "feature_metadata.json")):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        dim_rows.append(meta)
    write_csv(pd.DataFrame(dim_rows), os.path.join(outdir, "round13_feature_dim_summary.csv"))

    best = _best_row(aggregate)
    best_id = ""
    best_avg = np.nan
    if best is not None:
        best_id = str(best.get("Model_ID", best.get("ID", "")))
        best_avg = float(best.get("Average_TCGA_AUC_mean", np.nan))

    go_round14 = bool(best_avg > ROUND12_BEST)
    recommendation = "go_vicreg_stabilizer" if go_round14 else "round13_1_simplify_features"

    lines = [
        "# Round 13 Final Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Finetune jobs in manifest: {len(manifest)}",
        "",
        "## Downstream",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                f"- Best model: **{best_id}** — Avg TCGA **{best_avg:.4f}**",
                f"- vs Round 12 exp_037 ({ROUND12_BEST:.4f}): **{best_avg - ROUND12_BEST:+.4f}**",
                f"- vs Round 11 exp_035 ({ROUND11_BEST:.4f}): **{best_avg - ROUND11_BEST:+.4f}**",
                f"- vs R7 exp_048 ({R7_BEST:.4f}): **{best_avg - R7_BEST:+.4f}**",
                f"- Strong success threshold 0.6000: **{'met' if best_avg >= STRONG_SUCCESS else 'not met'}**",
            ]
        )
    else:
        lines.append("- Downstream aggregate not available.")

    if not feature_mode_df.empty:
        lines.extend(["", "## Feature mode summary", ""])
        for _, row in feature_mode_df.sort_values("best_avg_tcga", ascending=False).iterrows():
            lines.append(
                f"- `{row['prototype_feature_mode']}`: best {row['best_model_id']} "
                f"avg={row['best_avg_tcga']:.4f} (n={row['n_models']})"
            )

    if not delta_df.empty:
        improved = int((delta_df["delta_proto_minus_z_only"] > 0).sum())
        lines.extend(
            [
                "",
                "## z-only vs prototype features",
                "",
                f"- Models with proto > z-only: **{improved}/{len(delta_df)}**",
            ]
        )

    lines.extend(
        [
            "",
            "## Round 14 decision",
            "",
            f"**Recommendation:** `{recommendation}`",
            "",
        ]
    )
    if go_round14:
        lines.append(
            "Prototype-distance response features improved downstream beyond Round 12; "
            "consider low-weight VICReg / latent stabilizer re-integration (Round 14)."
        )
    else:
        lines.append(
            "Prototype features did not clearly beat Round 12 baseline; try Round 13.1 "
            "(own_cancer-only, robust scaler, stronger regularization) before Round 14."
        )

    report_path = os.path.join(outdir, "round13_final_report.md")
    write_md(report_path, lines)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 13 prototype response results")
    parser.add_argument("--run-dir", default="result/optimization_runs/round13_proto_response")
    parser.add_argument("--round12-root", default="result/optimization_runs/round12_proto_alignment")
    parser.add_argument("--round11-root", default="result/optimization_runs/round11_stability_recon")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    outdir = args.outdir or os.path.join(args.run_dir, "final_report")
    path = analyze_round13(
        args.run_dir,
        args.round12_root,
        args.round11_root,
        outdir,
        aggregate_path=args.aggregate,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
