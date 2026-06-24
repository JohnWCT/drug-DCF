#!/usr/bin/env python3
"""Analyze Round 14 VICReg stabilizer pretrain + downstream results."""

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

ROUND13_BEST = 0.6112395039184843
ROUND12_BEST = 0.5971789386885913
ROUND11_BEST = 0.5828
R7_BEST = 0.5918
STRONG_SUCCESS = 0.6200


def _vicreg_means_from_g_loss(exp_dir: str) -> dict:
    """Fallback: aggregate per-epoch VICReg losses from g_loss.csv."""
    g_loss_path = os.path.join(exp_dir, "g_loss.csv")
    if not os.path.isfile(g_loss_path):
        return {}
    try:
        df = pd.read_csv(g_loss_path)
    except Exception:
        return {}
    out: dict = {}
    for src_col, dst_col in (
        ("tumor_vicreg_var_loss", "tumor_vicreg_var_loss_mean"),
        ("tumor_vicreg_cov_loss", "tumor_vicreg_cov_loss_mean"),
    ):
        if src_col in df.columns:
            series = pd.to_numeric(df[src_col], errors="coerce").dropna()
            if not series.empty:
                out[dst_col] = float(series.mean())
    if "tumor_vicreg_var_loss_mean" in out and "tumor_vicreg_cov_loss_mean" in out:
        out["tumor_vicreg_loss_mean"] = (
            out["tumor_vicreg_var_loss_mean"] + out["tumor_vicreg_cov_loss_mean"]
        )
    return out


def _normalize_vicreg_row(row: dict, exp_dir: str) -> dict:
    """Map legacy per-step keys and g_loss.csv aggregates onto *_mean fields."""
    out = dict(row)
    for src, dst in (
        ("tumor_vicreg_var_loss", "tumor_vicreg_var_loss_mean"),
        ("tumor_vicreg_cov_loss", "tumor_vicreg_cov_loss_mean"),
    ):
        if pd.isna(out.get(dst)) and out.get(src) is not None:
            out[dst] = out[src]
    if pd.isna(out.get("latent_cov_offdiag_mean")) and out.get("tumor_vicreg_cov_offdiag_mean_abs") is not None:
        out["latent_cov_offdiag_mean"] = out["tumor_vicreg_cov_offdiag_mean_abs"]
    if pd.isna(out.get("tumor_vicreg_loss_mean")):
        var_m = out.get("tumor_vicreg_var_loss_mean")
        cov_m = out.get("tumor_vicreg_cov_loss_mean")
        if var_m is not None and cov_m is not None and not (pd.isna(var_m) or pd.isna(cov_m)):
            out["tumor_vicreg_loss_mean"] = float(var_m) + float(cov_m)
    fallback = _vicreg_means_from_g_loss(exp_dir)
    for key, value in fallback.items():
        if pd.isna(out.get(key)):
            out[key] = value
    return out


def _collect_pretrain_summaries(run_dir: str) -> pd.DataFrame:
    rows = []
    pretrain_dir = os.path.join(resolve_path(run_dir), "pretrain")
    for summary_path in sorted(glob.glob(os.path.join(pretrain_dir, "exp_*", "run_summary.json"))):
        exp_dir = os.path.dirname(summary_path)
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        params = payload.get("params", {})
        exp_id = payload.get("exp_id", os.path.basename(exp_dir))
        row = {"model_id": exp_id, **metrics}
        for key in (
            "round14_branch",
            "route_id",
            "source_model",
            "lambda_tumor_var",
            "lambda_tumor_cov",
            "tumor_vicreg_start_epoch",
            "tumor_vicreg_full_epoch",
            "conditional_adv_enabled",
            "source_anchor_proto_enabled",
            "lambda_proto_align",
            "reconstruction_loss_type",
            "latent_active_dims",
            "latent_cov_offdiag_mean",
            "tumor_vicreg_var_loss_mean",
            "tumor_vicreg_cov_loss_mean",
            "tumor_vicreg_loss_mean",
            "tumor_vicreg_var_loss",
            "tumor_vicreg_cov_loss",
            "tumor_vicreg_cov_offdiag_mean_abs",
            "mean_target_to_source_anchor_distance",
            "kmeans_ari",
            "wasserstein",
            "mean_conditional_leakage_strength",
            "random_seed",
        ):
            if key not in row and key in params:
                row[key] = params[key]
        rows.append(_normalize_vicreg_row(row, exp_dir))
    return pd.DataFrame(rows)


def _vicreg_sweep_summary(pretrain_df: pd.DataFrame) -> pd.DataFrame:
    if pretrain_df.empty:
        return pd.DataFrame()
    work = pretrain_df.copy()
    work["lambda_tumor_var"] = pd.to_numeric(work.get("lambda_tumor_var"), errors="coerce")
    work["lambda_tumor_cov"] = pd.to_numeric(work.get("lambda_tumor_cov"), errors="coerce")
    grouped = (
        work.groupby(["lambda_tumor_var", "lambda_tumor_cov", "round14_branch"], dropna=False)
        .agg(
            n_models=("model_id", "count"),
            mean_kmeans_ari=("kmeans_ari", "mean"),
            mean_active_dims=("latent_active_dims", "mean"),
            mean_cov_offdiag=("latent_cov_offdiag_mean", "mean"),
            mean_proto_gap=("mean_target_to_source_anchor_distance", "mean"),
        )
        .reset_index()
    )
    return grouped


def _latent_stability_summary(pretrain_df: pd.DataFrame) -> pd.DataFrame:
    if pretrain_df.empty:
        return pd.DataFrame()
    cols = [
        "model_id",
        "round14_branch",
        "route_id",
        "lambda_tumor_var",
        "lambda_tumor_cov",
        "latent_active_dims",
        "latent_cov_offdiag_mean",
        "kmeans_ari",
        "wasserstein",
        "tumor_vicreg_var_loss_mean",
        "tumor_vicreg_cov_loss_mean",
        "random_seed",
    ]
    keep = [c for c in cols if c in pretrain_df.columns]
    return pretrain_df[keep].copy()


def _infer_feature_mode(model_id: str) -> str:
    text = str(model_id)
    for suffix in ("own_plus_summary", "own_cancer", "none"):
        if text.endswith(suffix):
            return suffix
    return "none"


def _response_feature_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    id_col = "Model_ID" if "Model_ID" in aggregate.columns else "ID"
    work = aggregate.copy()
    if "prototype_feature_mode" not in work.columns:
        work["prototype_feature_mode"] = work[id_col].map(_infer_feature_mode)
    rows = []
    for mode, sub in work.groupby("prototype_feature_mode"):
        sub = sub.copy()
        sub["Average_TCGA_AUC_mean"] = pd.to_numeric(sub.get("Average_TCGA_AUC_mean"), errors="coerce")
        best = sub.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]
        rows.append(
            {
                "prototype_feature_mode": mode,
                "n_models": len(sub),
                "best_model_id": best.get(id_col),
                "best_avg_tcga": best.get("Average_TCGA_AUC_mean"),
                "mean_avg_tcga": sub["Average_TCGA_AUC_mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _z_vs_proto_delta(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    id_col = "Model_ID" if "Model_ID" in aggregate.columns else "ID"
    work = aggregate.copy()
    if "source_model_id" not in work.columns:
        work["source_model_id"] = work[id_col].astype(str).str.replace(r"^r14_", "", regex=True)
        for suffix in ("_own_plus_summary", "_own_cancer", "_none"):
            work["source_model_id"] = work["source_model_id"].str.replace(suffix, "", regex=False)
    if "prototype_feature_mode" not in work.columns:
        work["prototype_feature_mode"] = work[id_col].map(_infer_feature_mode)
    rows = []
    for source_id, sub in work.groupby("source_model_id"):
        z_only = sub[sub["prototype_feature_mode"].astype(str) == "none"]
        proto = sub[sub["prototype_feature_mode"].astype(str) != "none"]
        if z_only.empty or proto.empty:
            continue
        z_best = z_only.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]
        p_best = proto.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]
        rows.append(
            {
                "source_model_id": source_id,
                "z_only_model_id": z_best.get(id_col),
                "z_only_avg_tcga": z_best.get("Average_TCGA_AUC_mean"),
                "best_proto_model_id": p_best.get(id_col),
                "best_proto_feature_mode": p_best.get("prototype_feature_mode"),
                "best_proto_avg_tcga": p_best.get("Average_TCGA_AUC_mean"),
                "delta_proto_minus_z_only": float(p_best.get("Average_TCGA_AUC_mean", np.nan))
                - float(z_best.get("Average_TCGA_AUC_mean", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _route_comparison(aggregate: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    id_col = "Model_ID" if "Model_ID" in aggregate.columns else "ID"
    work = aggregate.copy()
    if not manifest.empty and "route_id" in manifest.columns:
        route_map = manifest.drop_duplicates("source_model_id").set_index("source_model_id")["route_id"]
        if "source_model_id" not in work.columns:
            work["source_model_id"] = work[id_col].astype(str)
        work["route_id"] = work["source_model_id"].map(route_map)
    rows = []
    for route, sub in work.groupby(work.get("route_id", pd.Series("unknown", index=work.index))):
        sub = sub.copy()
        sub["Average_TCGA_AUC_mean"] = pd.to_numeric(sub.get("Average_TCGA_AUC_mean"), errors="coerce")
        best = sub.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]
        rows.append(
            {
                "route_id": route,
                "n_models": len(sub),
                "best_model_id": best.get(id_col),
                "best_avg_tcga": best.get("Average_TCGA_AUC_mean"),
                "best_global_tcga": best.get("Global_TCGA_AUC_mean"),
            }
        )
    return pd.DataFrame(rows)


def analyze_round14(
    run_dir: str,
    round13_root: str,
    round12_root: str,
    outdir: str,
    aggregate_path: Optional[str] = None,
) -> str:
    run_dir = resolve_path(run_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    pretrain_summary = _collect_pretrain_summaries(run_dir)
    write_csv(pretrain_summary, os.path.join(outdir, "round14_pretrain_summary.csv"))
    write_csv(_vicreg_sweep_summary(pretrain_summary), os.path.join(outdir, "round14_vicreg_sweep_summary.csv"))
    write_csv(_latent_stability_summary(pretrain_summary), os.path.join(outdir, "round14_latent_stability_summary.csv"))

    agg_path = resolve_path(aggregate_path or os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    aggregate = pd.read_csv(agg_path) if os.path.isfile(agg_path) else pd.DataFrame()
    manifest_path = os.path.join(run_dir, "manifests", "finetune_dispatch_manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.isfile(manifest_path) else pd.DataFrame()

    if not aggregate.empty:
        ft_manifest = os.path.join(run_dir, "manifests", "finetune_dispatch_manifest.csv")
        if os.path.isfile(ft_manifest):
            ft = pd.read_csv(ft_manifest)
            merge_cols = [c for c in ("Model_ID", "source_model_id", "prototype_feature_mode", "route_id") if c in ft.columns]
            if merge_cols and "Model_ID" in aggregate.columns:
                id_map = ft.drop_duplicates("model_id").rename(columns={"model_id": "Model_ID"})
                for col in ("source_model_id", "prototype_feature_mode", "route_id", "lineage_source_model"):
                    if col in id_map.columns and col not in aggregate.columns:
                        aggregate = aggregate.merge(
                            id_map[["Model_ID", col]],
                            on="Model_ID",
                            how="left",
                        )

    write_csv(_response_feature_summary(aggregate), os.path.join(outdir, "round14_response_feature_summary.csv"))
    write_csv(_z_vs_proto_delta(aggregate), os.path.join(outdir, "round14_z_vs_proto_delta.csv"))
    write_csv(_route_comparison(aggregate, manifest), os.path.join(outdir, "round14_route_comparison.csv"))

    best_avg = np.nan
    best_model = ""
    if not aggregate.empty and "Average_TCGA_AUC_mean" in aggregate.columns:
        id_col = "Model_ID" if "Model_ID" in aggregate.columns else "ID"
        best_row = aggregate.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]
        best_avg = float(best_row["Average_TCGA_AUC_mean"])
        best_model = str(best_row.get(id_col, ""))

    seed_std = np.nan
    if not pretrain_summary.empty and "random_seed" in pretrain_summary.columns:
        if "sweetspot_tcga_proxy_score" in pretrain_summary.columns:
            seed_std = float(pretrain_summary.groupby("random_seed")["sweetspot_tcga_proxy_score"].mean().std())

    go_round15 = "hold"
    if pd.notna(best_avg):
        if best_avg > ROUND13_BEST:
            go_round15 = "go_importance_weighting"
        elif best_avg >= ROUND13_BEST - 0.003 and pd.notna(seed_std) and seed_std < 0.02:
            go_round15 = "go_importance_weighting_seed_stable"

    lines = [
        "# Round 14 Final Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Pretrain jobs in manifest: {len(pretrain_summary)}",
        "",
        "## Downstream",
        "",
        f"- Best model: **{best_model or 'pending'}** — Avg TCGA **{best_avg:.4f}**" if pd.notna(best_avg) else "- Best model: pending",
        f"- vs Round 13 best (0.6112): **{best_avg - ROUND13_BEST:+.4f}**" if pd.notna(best_avg) else "",
        f"- vs Round 12 exp_037 (0.5972): **{best_avg - ROUND12_BEST:+.4f}**" if pd.notna(best_avg) else "",
        f"- Strong success threshold 0.6200: **{'met' if pd.notna(best_avg) and best_avg >= STRONG_SUCCESS else 'not met'}**",
        "",
        "## Latent stability",
        "",
        f"- Pretrain summary rows: {len(pretrain_summary)}",
        f"- Seed proxy std (if available): {seed_std:.4f}" if pd.notna(seed_std) else "- Seed proxy std: n/a",
        "",
        "## Round 15 decision",
        "",
        f"**Recommendation:** `{go_round15}`",
        "",
    ]
    report_path = os.path.join(outdir, "round14_final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 14 VICReg stabilizer run")
    parser.add_argument("--run-dir", default="result/optimization_runs/round14_vicreg_stabilizer")
    parser.add_argument("--round13-root", default="result/optimization_runs/round13_proto_response")
    parser.add_argument("--round12-root", default="result/optimization_runs/round12_proto_alignment")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    outdir = args.outdir or os.path.join(args.run_dir, "final_report")
    analyze_round14(
        run_dir=args.run_dir,
        round13_root=args.round13_root,
        round12_root=args.round12_root,
        outdir=outdir,
        aggregate_path=args.aggregate,
    )


if __name__ == "__main__":
    main()
