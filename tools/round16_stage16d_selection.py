#!/usr/bin/env python3
"""Filter Stage 16D pretrain candidates for downstream finetune."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import resolve_path


def _load_pretrain_table(stage_root: str) -> pd.DataFrame:
    stage_root = resolve_path(stage_root)
    manifest_path = os.path.join(stage_root, "manifests", "stage16d_pretrain_manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "success"].copy()
    rows: List[dict] = []
    for _, row in manifest.iterrows():
        result_dir = resolve_path(str(row["result_dir"]))
        summary_path = os.path.join(result_dir, "run_summary.json")
        metrics = {}
        if os.path.isfile(summary_path):
            payload = json.load(open(summary_path, encoding="utf-8"))
            metrics = payload.get("metrics", {}) or {}
        lam_var = float(row.get("lambda_tumor_var", 0) or 0)
        lam_cov = float(row.get("lambda_tumor_cov", 0) or 0)
        rows.append(
            {
                **row.to_dict(),
                "pretrain_result_dir": os.path.relpath(result_dir, PROJECT_ROOT),
                "kmeans_ari": metrics.get("kmeans_ari"),
                "wasserstein": metrics.get("wasserstein"),
                "fid": metrics.get("fid"),
                "sweetspot_tcga_proxy_score": metrics.get(
                    "sweetspot_tcga_proxy_score", metrics.get("score_total")
                ),
                "vicreg_active": lam_var > 0 or lam_cov > 0,
                "lambda_sum": lam_var + lam_cov,
            }
        )
    return pd.DataFrame(rows)


def _score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["kmeans_ari"] = pd.to_numeric(work["kmeans_ari"], errors="coerce")
    work["wasserstein"] = pd.to_numeric(work["wasserstein"], errors="coerce")
    ari = work["kmeans_ari"]
    wass = work["wasserstein"]
    ari_n = (ari - ari.min()) / max(ari.max() - ari.min(), 1e-9)
    wass_n = 1.0 - (wass - wass.min()) / max(wass.max() - wass.min(), 1e-9)
    sweet = pd.to_numeric(work.get("sweetspot_tcga_proxy_score"), errors="coerce").fillna(0.0)
    sweet_n = (sweet - sweet.min()) / max(sweet.max() - sweet.min(), 1e-9) if sweet.notna().any() else 0.0
    work["round16d_pretrain_score"] = 0.50 * ari_n.fillna(0) + 0.35 * wass_n.fillna(0) + 0.15 * sweet_n
    return work


def _pick_lineage(pool: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Pick diverse unique checkpoints per lineage."""
    if pool.empty:
        return pool

    work = pool.copy()
    work["selection_reason"] = ""
    chosen_parts = []

    def _best_unique(sub: pd.DataFrame, reason: str, n: int = 1) -> pd.DataFrame:
        if sub.empty:
            return sub
        picked = (
            sub.sort_values("round16d_pretrain_score", ascending=False)
            .drop_duplicates(subset=["pretrain_result_dir"], keep="first")
            .head(n)
            .copy()
        )
        picked["selection_reason"] = reason
        return picked

    no_vic = work[(work["lambda_tumor_var"].fillna(0) == 0) & (work["lambda_tumor_cov"].fillna(0) == 0)]
    chosen_parts.append(_best_unique(no_vic, "no_vicreg_control", 1))

    used_dirs = set(chosen_parts[-1]["pretrain_result_dir"]) if not chosen_parts[-1].empty else set()
    remain = work[~work["pretrain_result_dir"].isin(used_dirs)].copy()

    low = remain[remain["vicreg_active"] & (remain["lambda_sum"] <= 2e-6)]
    picked_low = _best_unique(low, "ultra_low_vicreg", 1)
    used_dirs.update(picked_low["pretrain_result_dir"])
    chosen_parts.append(picked_low)

    remain = work[~work["pretrain_result_dir"].isin(used_dirs)].copy()
    high = remain[remain["vicreg_active"] & (remain["lambda_sum"] > 2e-6)]
    picked_high = _best_unique(high, "higher_vicreg", 1)
    used_dirs.update(picked_high["pretrain_result_dir"])
    chosen_parts.append(picked_high)

    remain = work[~work["pretrain_result_dir"].isin(used_dirs)].copy()
    slots = max(0, top_k - sum(len(p) for p in chosen_parts if not p.empty))
    if slots > 0:
        picked_top = _best_unique(remain, "top_pretrain_score", slots)
        chosen_parts.append(picked_top)

    out = pd.concat([p for p in chosen_parts if not p.empty], ignore_index=True)
    out["selected"] = True
    return out.drop_duplicates(subset=["pretrain_result_dir"], keep="first").head(top_k)


def select_stage16d_candidates(
    stage_root: str,
    *,
    top_k_per_lineage: int = 4,
    outdir: str | None = None,
) -> pd.DataFrame:
    stage_root = resolve_path(stage_root)
    outdir = resolve_path(outdir or os.path.join(stage_root, "reports"))
    os.makedirs(outdir, exist_ok=True)

    table = _load_pretrain_table(stage_root)
    if table.empty:
        raise ValueError("No successful 16D pretrain jobs found")

    scored = _score_candidates(table)
    selected_parts = []
    for lineage, sub in scored.groupby("round16_lineage"):
        picked = _pick_lineage(sub, top_k_per_lineage)
        picked["round16_lineage"] = lineage
        selected_parts.append(picked)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    selected["downstream_model_id"] = selected.apply(
        lambda r: f"r16d_{r['round16_lineage']}_{r['exp_id']}", axis=1
    )

    all_scored = scored.copy()
    all_scored["selected"] = all_scored["exp_id"].isin(selected["exp_id"])
    all_scored["selection_reason"] = ""
    if not selected.empty and "selection_reason" in selected.columns:
        reason_map = selected.set_index("exp_id")["selection_reason"].to_dict()
        all_scored["selection_reason"] = all_scored["exp_id"].map(reason_map).fillna("")

    all_path = os.path.join(outdir, "stage16d_pretrain_scored.csv")
    sel_path = os.path.join(outdir, "stage16d_pretrain_candidates.csv")
    all_scored.sort_values(["round16_lineage", "round16d_pretrain_score"], ascending=[True, False]).to_csv(
        all_path, index=False
    )
    selected.to_csv(sel_path, index=False)
    print(f"Wrote {len(all_scored)} scored rows -> {all_path}")
    print(f"Selected {len(selected)} candidates -> {sel_path}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Stage 16D pretrain candidates")
    parser.add_argument(
        "--stage-root",
        default="result/optimization_runs/round16_bruteforce/stage16d",
    )
    parser.add_argument("--top-k-per-lineage", type=int, default=4)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    select_stage16d_candidates(
        args.stage_root,
        top_k_per_lineage=args.top_k_per_lineage,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
