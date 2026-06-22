#!/usr/bin/env python3
"""Round 9 latent stability diagnostics."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import (
    effective_rank,
    iter_reproduction_models,
    latent_matrix_and_labels,
    load_latent_domain_frame,
    write_csv,
    write_md,
)


def analyze_model(model: dict) -> tuple[dict, dict, dict]:
    frame = load_latent_domain_frame(model["checkpoint_dir"])
    x, _, _ = latent_matrix_and_labels(frame)
    latent_size = x.shape[1]
    stds = x.std(axis=0)
    active_001 = int((stds > 0.01).sum())
    active_005 = int((stds > 0.05).sum())
    cov = np.cov(x, rowvar=False)
    off_diag = cov - np.diag(np.diag(cov))
    eff_rank = effective_rank(x)
    collapse_flag = active_001 < 0.5 * latent_size
    redundancy_flag = float(np.mean(np.abs(off_diag))) > 0.25 if off_diag.size else False
    model_row = {
        "source_exp_id": model.get("source_exp_id", ""),
        "role": model.get("role", ""),
        "reproduction_seed": model.get("reproduction_seed", ""),
        "model_id": model.get("model_id", ""),
        "latent_size": latent_size,
        "active_dim_count_std_gt_0_01": active_001,
        "active_dim_count_std_gt_0_05": active_005,
        "mean_latent_std": float(stds.mean()),
        "median_latent_std": float(np.median(stds)),
        "min_latent_std": float(stds.min()),
        "max_latent_std": float(stds.max()),
        "mean_abs_offdiag_cov": float(np.mean(np.abs(off_diag))) if off_diag.size else float("nan"),
        "max_abs_offdiag_cov": float(np.max(np.abs(off_diag))) if off_diag.size else float("nan"),
        "effective_rank": eff_rank,
        "collapse_flag": bool(collapse_flag),
        "redundancy_flag": bool(redundancy_flag),
        "notes": "",
    }
    dim_rows = [{"model_id": model.get("model_id", ""), "dim": i, "latent_std": float(stds[i])} for i in range(latent_size)]
    cov_row = {
        "model_id": model.get("model_id", ""),
        "mean_abs_offdiag_cov": model_row["mean_abs_offdiag_cov"],
        "max_abs_offdiag_cov": model_row["max_abs_offdiag_cov"],
        "effective_rank": eff_rank,
    }
    return model_row, dim_rows, cov_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent stability diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--latent-view", default="shared")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    model_rows = []
    dim_rows = []
    cov_rows = []
    for model in iter_reproduction_models(args.run_dir):
        try:
            model_row, dims, cov_row = analyze_model(model)
            model_rows.append(model_row)
            dim_rows.extend(dims)
            cov_rows.append(cov_row)
        except Exception as exc:
            model_rows.append({"model_id": model.get("model_id", ""), "notes": str(exc)})

    write_csv(pd.DataFrame(model_rows), os.path.join(args.outdir, "latent_stability_by_model.csv"))
    write_csv(pd.DataFrame(dim_rows), os.path.join(args.outdir, "latent_dimension_stats.csv"))
    write_csv(pd.DataFrame(cov_rows), os.path.join(args.outdir, "latent_covariance_stats.csv"))
    write_md(os.path.join(args.outdir, "latent_stability_report.md"), ["# Latent Stability", ""])
    print(f"Wrote {args.outdir}")


if __name__ == "__main__":
    main()
