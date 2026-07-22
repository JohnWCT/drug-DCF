#!/usr/bin/env python3
"""Aggregate Round23 paired performance and write CSV + selection inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.validation.model_comparison import paired_model_deltas, summarize_paired_deltas
from tools.biocda_telegram_notify import biocda_notify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/biocda/xa_v2_closure.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = ROOT / config["outputs"]["root"]
    rows = []
    for seed in config["experiment"]["seeds"]:
        for model in list(config["models"]) + ["biocda_xa_z_only"]:
            run_dir = out_root / f"{model}_seed{seed}"
            metrics_path = run_dir / "metrics_by_seed.json"
            if not metrics_path.is_file():
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            vm = payload["validation"]
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "drug_macro_auc": vm.get("DrugMacro_AUC"),
                    "drug_macro_auprc": vm.get("DrugMacro_AUPRC"),
                    "sample_auc": vm.get("Global_AUC"),
                    "sample_auprc": vm.get("Global_AUPRC"),
                    "best_epoch": payload.get("best_epoch"),
                }
            )
    df = pd.DataFrame(rows)
    reports = ROOT / config["outputs"]["reports_root"]
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "round23_paired_performance.csv"
    df.to_csv(csv_path, index=False)

    if not df.empty and "biocda_predictive" in set(df["model"]):
        pairs = []
        for cand in ("biocda_xa_fresh", "biocda_xa_transfer", "biocda_xa_kd", "biocda_xa_z_only"):
            if cand in set(df["model"]):
                pairs.append((cand, "biocda_predictive"))
        if pairs:
            d = paired_model_deltas(
                df,
                metric_columns=["drug_macro_auc", "drug_macro_auprc"],
                pairs=pairs,
            )
            summary = summarize_paired_deltas(d)
            (reports / "round23_paired_deltas.json").write_text(
                summary.to_json(orient="records", indent=2) + "\n",
                encoding="utf-8",
            )

    biocda_notify(f"Round23 evaluate wrote {csv_path} n={len(df)}")
    print(df.to_string(index=False) if len(df) else "NO_METRICS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
