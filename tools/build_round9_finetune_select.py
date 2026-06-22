#!/usr/bin/env python3
"""Build Round 9 finetune model_select from reproduction checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import (
    iter_reproduction_models,
    load_exp_metrics,
    resolve_path,
    write_csv,
)


def build_round9_finetune_select(
    run_dir: str,
    resolved_baselines_path: str,
    include_all_reproductions: bool = True,
) -> pd.DataFrame:
    resolved = pd.read_csv(resolve_path(resolved_baselines_path))
    resolved_ids = set(resolved[resolved["resolved"] == True]["exp_id"].astype(str))  # noqa: E712
    models = iter_reproduction_models(run_dir)
    rows: List[dict] = []
    seen_ids = set()
    for model in models:
        source_exp = str(model.get("source_exp_id", ""))
        if source_exp and source_exp not in resolved_ids:
            continue
        model_id = str(model.get("model_id", ""))
        if not model_id or model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        checkpoint_dir = model["checkpoint_dir"]
        metrics = load_exp_metrics(checkpoint_dir)
        params_path = os.path.join(checkpoint_dir, "params.json")
        params = {}
        if os.path.exists(params_path):
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f).get("params", {})
        rows.append(
            {
                "ID": model_id,
                "NO": len(rows) + 1,
                "model_type": "VAEwC",
                "source_exp_id": source_exp,
                "source_role": model.get("role", ""),
                "reproduction_seed": model.get("reproduction_seed", ""),
                "pretrain_epochs": params.get("pretrain_num_epochs", ""),
                "train_epochs": params.get("train_num_epochs", ""),
                "pretrain_lr": params.get("pretrain_learning_rate", ""),
                "train_lr": params.get("gan_learning_rate", ""),
                "dropout": params.get("dropout_rate", ""),
                "encoder_dims": str(params.get("encoder_dims", "")),
                "lambda_cls": params.get("lambda_cls", ""),
                "use_class_weight": params.get("use_class_weight", ""),
                "fid": metrics.get("fid", ""),
                "wasserstein": metrics.get("wasserstein", ""),
                "kmeans_ari": metrics.get("kmeans_ari", ""),
                "kmeans_nmi": metrics.get("kmeans_nmi", ""),
                "selection_rank": len(rows) + 1,
                "result_folder": os.path.relpath(checkpoint_dir, PROJECT_ROOT),
                "pretrain_result_dir": os.path.relpath(checkpoint_dir, PROJECT_ROOT),
            }
        )
    if not rows:
        raise RuntimeError(
            "No successful Round 9 reproduction checkpoints found; "
            "cannot build finetune model_select."
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 9 finetune model_select")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resolved-baselines", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--include-all-reproductions", action="store_true", default=True)
    args = parser.parse_args()

    outdir = resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    df = build_round9_finetune_select(
        args.run_dir,
        args.resolved_baselines,
        include_all_reproductions=args.include_all_reproductions,
    )
    path = write_csv(df, os.path.join(outdir, "model_select.csv"))
    print(f"Wrote {path} ({len(df)} models)")


if __name__ == "__main__":
    main()
