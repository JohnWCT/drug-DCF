"""Stage 24D: gdsc_intersect13 per-drug diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from biocda.validation.round24_protocol import (
    PROJECT_ROOT,
    metrics_from_predictions,
    prepare_tcga_with_drops,
)

ROOT = PROJECT_ROOT


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "DRUG_NAME" not in df.columns and "drug_name" in df.columns:
        df["DRUG_NAME"] = df["drug_name"]
    return df


def run_diagnose(cfg: Dict[str, Any], out_dir: Path, *, target: str = "gdsc_intersect13") -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = str(ROOT / cfg["baseline"]["feature_dir"])
    smiles = str(ROOT / cfg["paths"]["drug_smiles"])
    tpath = next(t["path"] for t in cfg["targets"] if t["key"] == target)
    kept, dropped, meta = prepare_tcga_with_drops(
        str(ROOT / tpath),
        feature_dir=feature_dir,
        drug_smiles_path=smiles,
        target_key=target,
    )
    dropped.to_csv(out_dir / "coverage_and_support.csv", index=False)

    # Sources: R18 baseline ensemble + R23 P0/X0 ensembles if present
    sources = {
        "B0_r18_own_plus_summary": ROOT
        / cfg["paths"]["round18_root"]
        / "reports"
        / f"round18e_tcga_pooled_mlp__own_plus_summary_{target}_ensemble_predictions.csv",
        "P0_r23": ROOT / "reports/biocda_tcga_comparison/r23_biocda_predictive" / f"predictions_ensemble_{target}.csv",
        "X0_r23": ROOT / "reports/biocda_tcga_comparison/r23_biocda_xa_fresh" / f"predictions_ensemble_{target}.csv",
    }

    per_drug_rows: List[Dict[str, Any]] = []
    weakness = []
    calib_rows = []
    summary = {"target": target, "cohort": meta, "models": {}}

    for name, path in sources.items():
        if not path.is_file():
            summary["models"][name] = {"status": "missing", "path": str(path)}
            continue
        df = _load_pred(path)
        m = metrics_from_predictions(df, cfg)
        summary["models"][name] = {
            "status": "ok",
            "DrugMacro_AUC": m["DrugMacro_AUC"],
            "Global_AUC": m["Global_AUC"],
            "n_rows": m["n_rows"],
            "n_valid_auc_drugs": m["n_valid_auc_drugs"],
            "gap_global_minus_drugmacro": (
                None
                if m["Global_AUC"] is None or m["DrugMacro_AUC"] is None
                else float(m["Global_AUC"] - m["DrugMacro_AUC"])
            ),
        }
        per = m.get("per_drug")
        if per is None:
            continue
        if isinstance(per, pd.DataFrame):
            pdf = per.copy()
        else:
            pdf = pd.DataFrame(per)
        pdf["model"] = name
        per_drug_rows.append(pdf)
        # calibration
        y = df["Label"].astype(int).to_numpy()
        p = df["probability"].astype(float).to_numpy()
        try:
            brier = float(brier_score_loss(y, p))
        except Exception:
            brier = None
        calib_rows.append({"model": name, "brier": brier, "n": len(df)})
        if "AUC" in pdf.columns:
            bottom = pdf.nsmallest(5, "AUC")
            for _, r in bottom.iterrows():
                weakness.append(
                    {
                        "model": name,
                        "DRUG_NAME": r.get("drug") or r.get("DRUG_NAME"),
                        "AUC": r.get("AUC"),
                        "AUPRC": r.get("AUPRC"),
                        "n_samples": r.get("n_samples"),
                    }
                )

    if per_drug_rows:
        all_pd = pd.concat(per_drug_rows, ignore_index=True)
        all_pd.to_csv(out_dir / "gdsc_intersect13_per_drug.csv", index=False)
    else:
        all_pd = pd.DataFrame()
    pd.DataFrame(weakness).to_csv(out_dir / "weakness_overlap.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration_summary.csv", index=False)

    md = [
        f"# gdsc_intersect13 diagnostic ({target})",
        "",
        f"- raw={meta['n_raw']} eligible={meta['n_eligible']} dropped={meta['n_dropped']}",
        f"- miss_latent={meta['n_miss_latent']} miss_smiles={meta['n_miss_smiles']}",
        "",
        "## Model summary",
        "",
    ]
    for name, info in summary["models"].items():
        md.append(f"- **{name}**: {info}")
    md += ["", "## Notes", "", "- Calibration is diagnostic only; not used for selection.", ""]
    (out_dir / "gdsc_intersect13_diagnostic.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
