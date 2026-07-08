#!/usr/bin/env python3
"""Round 9 cancer prototype diagnostics."""

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

from tools.round9_diagnostics_common import (
    cosine_distance,
    euclidean_distance,
    iter_reproduction_models,
    latent_matrix_and_labels,
    load_latent_domain_frame,
    macro_mean,
    write_csv,
    write_md,
)


def _prototype(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.array([])
    return vectors.mean(axis=0)


def analyze_model(
    model: dict,
    metrics: List[str],
    min_source: int,
    min_target: int,
    frame: Optional[pd.DataFrame] = None,
):
    if frame is None:
        frame = load_latent_domain_frame(model["checkpoint_dir"])
    z_cols = [c for c in frame.columns if c.startswith("z")]
    by_cancer_rows = []
    source_vectors = []
    target_vectors = []
    for cancer_type, sub in frame.groupby("cancer_type"):
        source = sub[sub["domain"] == "source"][z_cols].to_numpy(dtype=np.float32)
        target = sub[sub["domain"] == "target"][z_cols].to_numpy(dtype=np.float32)
        n_source = len(source)
        n_target = len(target)
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
            "notes": "" if sufficient else "insufficient_samples",
        }
        if sufficient:
            src_proto = _prototype(source)
            tgt_proto = _prototype(target)
            if "cosine" in metrics:
                row["source_target_cosine_distance"] = cosine_distance(src_proto, tgt_proto)
            if "euclidean" in metrics:
                row["source_target_euclidean_distance"] = euclidean_distance(src_proto, tgt_proto)
            row["source_proto_norm"] = float(np.linalg.norm(src_proto))
            row["target_proto_norm"] = float(np.linalg.norm(tgt_proto))
            source_vectors.append((cancer_type, src_proto))
            target_vectors.append((cancer_type, tgt_proto))
        by_cancer_rows.append(row)

    def inter_margin(vectors):
        if len(vectors) < 2:
            return float("nan")
        margins = []
        for i, (_, a) in enumerate(vectors):
            others = [b for j, (_, b) in enumerate(vectors) if j != i]
            dists = [euclidean_distance(a, b) for b in others]
            margins.append(min(dists))
        return macro_mean(margins)

    sufficient_rows = [r for r in by_cancer_rows if r.get("sufficient_samples")]
    summary = {
        "source_exp_id": model.get("source_exp_id", ""),
        "role": model.get("role", ""),
        "reproduction_seed": model.get("reproduction_seed", ""),
        "model_id": model.get("model_id", ""),
        "mean_same_cancer_source_target_cosine_distance": macro_mean([r.get("source_target_cosine_distance", np.nan) for r in sufficient_rows]),
        "mean_same_cancer_source_target_euclidean_distance": macro_mean([r.get("source_target_euclidean_distance", np.nan) for r in sufficient_rows]),
        "mean_inter_cancer_source_margin": inter_margin(source_vectors),
        "mean_inter_cancer_target_margin": inter_margin(target_vectors),
        "prototype_alignment_ratio": float("nan"),
        "worst_aligned_cancer_type": "",
        "best_aligned_cancer_type": "",
        "n_sufficient_cancer_types": len(sufficient_rows),
    }
    if sufficient_rows:
        worst = max(sufficient_rows, key=lambda r: r.get("source_target_cosine_distance", -1))
        best = min(sufficient_rows, key=lambda r: r.get("source_target_cosine_distance", 1e9))
        summary["worst_aligned_cancer_type"] = worst["cancer_type"]
        summary["best_aligned_cancer_type"] = best["cancer_type"]
        inter = summary["mean_inter_cancer_source_margin"]
        same = summary["mean_same_cancer_source_target_euclidean_distance"]
        if inter and not np.isnan(inter) and inter > 0:
            summary["prototype_alignment_ratio"] = same / inter if not np.isnan(same) else float("nan")

    src_df = pd.DataFrame(
        [{"cancer_type": c, **{f"z{i}": v[i] for i in range(len(v))}} for c, v in source_vectors]
    )
    tgt_df = pd.DataFrame(
        [{"cancer_type": c, **{f"z{i}": v[i] for i in range(len(v))}} for c, v in target_vectors]
    )
    return pd.DataFrame(by_cancer_rows), summary, src_df, tgt_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancer prototype diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--latent-view", default="shared")
    parser.add_argument("--min-source-per-cancer", type=int, default=10)
    parser.add_argument("--min-target-per-cancer", type=int, default=10)
    parser.add_argument("--metrics", nargs="+", default=["cosine", "euclidean"])
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    by_cancer_all = []
    summaries = []
    source_all = []
    target_all = []
    for model in iter_reproduction_models(args.run_dir):
        try:
            by_cancer, summary, src_df, tgt_df = analyze_model(
                model, args.metrics, args.min_source_per_cancer, args.min_target_per_cancer
            )
            by_cancer["model_id"] = model.get("model_id", "")
            by_cancer_all.append(by_cancer)
            summaries.append(summary)
            if not src_df.empty:
                src_df.insert(0, "model_id", model.get("model_id", ""))
                source_all.append(src_df)
            if not tgt_df.empty:
                tgt_df.insert(0, "model_id", model.get("model_id", ""))
                target_all.append(tgt_df)
        except Exception as exc:
            summaries.append({"model_id": model.get("model_id", ""), "notes": str(exc)})

    write_csv(pd.concat(by_cancer_all, ignore_index=True) if by_cancer_all else pd.DataFrame(), os.path.join(args.outdir, "prototype_distance_by_cancer.csv"))
    write_csv(pd.DataFrame(summaries), os.path.join(args.outdir, "prototype_margin_summary.csv"))
    write_csv(pd.concat(source_all, ignore_index=True) if source_all else pd.DataFrame(), os.path.join(args.outdir, "prototype_vectors_source.csv"))
    write_csv(pd.concat(target_all, ignore_index=True) if target_all else pd.DataFrame(), os.path.join(args.outdir, "prototype_vectors_target.csv"))
    write_md(os.path.join(args.outdir, "prototype_diagnostics_report.md"), ["# Prototype Diagnostics", ""])
    print(f"Wrote {args.outdir}")


if __name__ == "__main__":
    main()
