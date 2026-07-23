"""Stage 24G: GDSC vs TCGA objective alignment (diagnostic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]


def run_alignment(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gdsc_path = ROOT / "reports/round23_paired_performance.csv"
    tcga_path = ROOT / "reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv"
    report: Dict[str, Any] = {"status": "ok", "spearman": {}, "notes": []}
    if not gdsc_path.is_file() or not tcga_path.is_file():
        report["status"] = "missing_inputs"
        report["notes"].append("Need round23_paired_performance.csv and biocda_tcga_comparison_long.csv")
        (out_dir / "objective_alignment.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    gdsc = pd.read_csv(gdsc_path)
    tcga = pd.read_csv(tcga_path)
    # Aggregate TCGA mean DrugMacro per model
    if "mean_DrugMacro_AUC_5targets" in tcga.columns:
        tcga_mean = tcga.groupby("model_id")["mean_DrugMacro_AUC_5targets"].first()
    else:
        tcga_mean = tcga.groupby("model_id")["DrugMacro_AUC"].mean()

    # Best-effort join on model name fragments
    pairs = []
    for _, row in gdsc.iterrows():
        mid = str(row.get("model") or row.get("model_id") or "")
        # find matching tcga model
        matches = [k for k in tcga_mean.index if mid and mid in str(k)]
        if not matches:
            continue
        pairs.append((float(row.get("mean_auc") or row.get("DrugMacro_AUC") or row.iloc[1]), float(tcga_mean[matches[0]])))

    if len(pairs) >= 3:
        xs, ys = zip(*pairs)
        corr = spearmanr(xs, ys)
        report["spearman"] = {"rho": float(corr.correlation), "pvalue": float(corr.pvalue), "n": len(pairs)}
    else:
        report["notes"].append(f"insufficient paired models: {len(pairs)}")
    report["selection_role_gdsc"] = "none"
    report["claim"] = "Association only; not causal. GDSC not used for Round24 lock."
    (out_dir / "objective_alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
