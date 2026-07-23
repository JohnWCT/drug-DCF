"""Round 24 eval3 protocol: cohort lock, DrugMacro, baseline rebuild."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics
from tools.round18_tcga_dataset import (
    build_patient_to_latent_key,
    load_tcga_omics_latent_dict,
    make_eval_row_id,
)
from tools.finetune_tcga_eval import load_tcga_response_csv
from tools.round18_eligible_data import load_smiles_lookup

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval3_config(path: Path | str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    cfg["_config_sha256"] = sha256_file(cfg_path)
    return cfg


def gate_table(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out = {}
    for t in cfg["targets"]:
        out[t["key"]] = {
            "gate_auroc": float(t["gate_auroc"]),
            "gate_auprc": float(t["gate_auprc"]),
            "gate_auroc_std": float(t["gate_auroc_std"]),
            "headline_n_pairs": int(t["headline_n_pairs"]),
            "path": t["path"],
        }
    return out


def prepare_tcga_with_drops(
    response_path: str,
    *,
    feature_dir: str,
    drug_smiles_path: str,
    target_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Like Round18 prepare_tcga_response_frame, but also returns dropped_rows."""
    raw = load_tcga_response_csv(response_path).copy()
    drug_col = "drug_name" if "drug_name" in raw.columns else "DRUG_NAME"
    raw = raw.reset_index(drop=True)
    raw["_source_row_id"] = np.arange(len(raw), dtype=int)
    raw["Patient_id"] = raw["Patient_id"].astype(str)
    raw["DRUG_NAME"] = raw[drug_col].astype(str).str.strip()
    raw["Label"] = raw["Label"].astype(int)

    tcga_latent = load_tcga_omics_latent_dict(feature_dir)
    patient_map = build_patient_to_latent_key(tcga_latent)
    smiles_lookup = load_smiles_lookup(drug_smiles_path)

    kept_rows: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []
    n_miss_latent = 0
    n_miss_smiles = 0
    for _, row in raw.iterrows():
        pid = str(row["Patient_id"])
        drug = str(row["DRUG_NAME"]).strip()
        label = int(row["Label"])
        sid = int(row["_source_row_id"])
        if pid not in patient_map:
            n_miss_latent += 1
            dropped_rows.append(
                {
                    "target_key": target_key,
                    "source_row_id": sid,
                    "Patient_id": pid,
                    "DRUG_NAME": drug,
                    "Label": label,
                    "drop_reason": "miss_latent",
                    "cancers": row.get("cancers", None),
                }
            )
            continue
        smiles = None
        if "smiles" in row.index and pd.notna(row["smiles"]) and str(row["smiles"]).strip():
            smiles = str(row["smiles"]).strip()
        elif drug.lower() in smiles_lookup:
            smiles = smiles_lookup[drug.lower()]
        else:
            n_miss_smiles += 1
            dropped_rows.append(
                {
                    "target_key": target_key,
                    "source_row_id": sid,
                    "Patient_id": pid,
                    "DRUG_NAME": drug,
                    "Label": label,
                    "drop_reason": "miss_smiles",
                    "cancers": row.get("cancers", None),
                }
            )
            continue
        kept_rows.append(
            {
                "target_key": target_key,
                "_source_row_id": sid,
                "Patient_id": pid,
                "DRUG_NAME": drug,
                "Label": label,
                "smiles": smiles,
                "latent_key": patient_map[pid],
                "eval_row_id": make_eval_row_id(
                    target_key=target_key,
                    patient_id=pid,
                    drug_name=drug,
                    label=label,
                    source_row_id=sid,
                ),
                "cancers": row.get("cancers", None),
            }
        )

    kept = pd.DataFrame(kept_rows)
    dropped = pd.DataFrame(dropped_rows)
    feature_pkl = Path(feature_dir) / "tcga_latent_proto.pkl"
    meta = {
        "target_key": target_key,
        "response_path": response_path,
        "feature_dir": feature_dir,
        "n_raw": int(len(raw)),
        "n_eligible": int(len(kept)),
        "n_dropped": int(len(dropped)),
        "n_miss_latent": int(n_miss_latent),
        "n_miss_smiles": int(n_miss_smiles),
        "n_latent_keys": int(len(tcga_latent)),
        "n_patient_map": int(len(patient_map)),
        "csv_sha256": sha256_file(PROJECT_ROOT / response_path if not Path(response_path).is_absolute() else Path(response_path)),
        "feature_sha256": sha256_file(feature_pkl) if feature_pkl.is_file() else None,
    }
    return kept, dropped, meta


def metrics_from_predictions(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    dm = cfg["drug_macro"]
    work = df.copy()
    if "DRUG_NAME" not in work.columns and "drug_name" in work.columns:
        work["DRUG_NAME"] = work["drug_name"]
    m = calculate_robust_drug_macro_metrics(
        work,
        min_samples=int(dm["min_samples"]),
        min_positive=int(dm["min_positive"]),
        min_negative=int(dm["min_negative"]),
    )
    return {
        "DrugMacro_AUC": m.get("DrugMacro_AUC"),
        "DrugMacro_AUPRC": m.get("DrugMacro_AUPRC"),
        "Global_AUC": m.get("Global_AUC"),
        "Global_AUPRC": m.get("Global_AUPRC"),
        "n_valid_auc_drugs": m.get("n_valid_auc_drugs"),
        "n_rows": int(len(work)),
        "per_drug": m.get("per_drug"),
    }


def average_tcga_auc_proxy(df: pd.DataFrame) -> Optional[float]:
    """Historical Average_TCGA_AUC ≈ mean of per-drug AUC without support filter."""
    from sklearn.metrics import roc_auc_score

    work = df.copy()
    if "DRUG_NAME" not in work.columns and "drug_name" in work.columns:
        work["DRUG_NAME"] = work["drug_name"]
    vals = []
    for _, g in work.groupby("DRUG_NAME", sort=False):
        y = g["Label"].to_numpy().astype(int)
        p = g["probability"].to_numpy().astype(float)
        if len(np.unique(y)) < 2:
            continue
        try:
            vals.append(float(roc_auc_score(y, p)))
        except ValueError:
            continue
    return float(np.mean(vals)) if vals else None


def run_preflight(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = str(PROJECT_ROOT / cfg["baseline"]["feature_dir"])
    smiles = str(PROJECT_ROOT / cfg["paths"]["drug_smiles"])
    coverage_rows = []
    dropped_all = []
    target_meta = {}
    blockers: List[str] = []

    for t in cfg["targets"]:
        key = t["key"]
        path = str(PROJECT_ROOT / t["path"])
        kept, dropped, meta = prepare_tcga_with_drops(
            path,
            feature_dir=feature_dir,
            drug_smiles_path=smiles,
            target_key=key,
        )
        target_meta[key] = meta
        coverage_rows.append(
            {
                "target_key": key,
                "headline_n_pairs": int(t["headline_n_pairs"]),
                "n_raw": meta["n_raw"],
                "n_eligible": meta["n_eligible"],
                "n_dropped": meta["n_dropped"],
                "n_miss_latent": meta["n_miss_latent"],
                "n_miss_smiles": meta["n_miss_smiles"],
                "raw_matches_headline": meta["n_raw"] == int(t["headline_n_pairs"]),
                "csv_sha256": meta["csv_sha256"],
            }
        )
        if not dropped.empty:
            dropped_all.append(dropped)
        if meta["n_raw"] != int(t["headline_n_pairs"]):
            blockers.append(
                f"{key}: raw={meta['n_raw']} != headline={t['headline_n_pairs']}"
            )
        if meta["n_dropped"] and dropped.empty:
            blockers.append(f"{key}: dropped count>0 but dropped_rows empty")
        if not dropped.empty and dropped["drop_reason"].isna().any():
            blockers.append(f"{key}: missing drop_reason")

    coverage = pd.DataFrame(coverage_rows)
    dropped_df = pd.concat(dropped_all, ignore_index=True) if dropped_all else pd.DataFrame()
    coverage.to_csv(out_dir / "cohort_coverage.csv", index=False)
    dropped_df.to_csv(out_dir / "dropped_rows.csv", index=False)

    g = target_meta["gdsc_intersect13"]
    if g["n_raw"] != 906 or g["n_eligible"] != 886:
        blockers.append(
            f"gdsc_intersect13 expected raw=906 eligible=886, got raw={g['n_raw']} eligible={g['n_eligible']}"
        )

    r18_root = PROJECT_ROOT / cfg["paths"]["round18_root"]
    fold_assign = r18_root / "splits" / "formal_5fold_assignments.csv"
    if not fold_assign.is_file():
        blockers.append(f"missing Round18 folds: {fold_assign}")
    else:
        folds = pd.read_csv(fold_assign)
        n_folds = int(folds["fold_id"].nunique()) if "fold_id" in folds.columns else 0
        if n_folds != int(cfg["n_folds"]):
            blockers.append(f"expected {cfg['n_folds']} folds, got {n_folds}")

    manifest = {
        "protocol_name": cfg["protocol_name"],
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "n_folds": cfg["n_folds"],
        "drug_macro": cfg["drug_macro"],
        "target_priority": cfg["target_priority"],
        "target_weights": cfg["target_weights"],
        "baseline": cfg["baseline"],
        "feature_dir": feature_dir,
        "drug_smiles": smiles,
        "targets": target_meta,
        "cohort_coverage": coverage_rows,
        "gate_table": gate_table(cfg),
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "gdsc_intersect13_note": {
            "headline_pairs": 906,
            "eligible_pairs": g["n_eligible"],
            "dropped": g["n_dropped"],
            "drop_breakdown": {
                "miss_latent": g["n_miss_latent"],
                "miss_smiles": g["n_miss_smiles"],
            },
            "explanation": (
                "906 raw TCGA pairs minus patients missing from tcga_latent_proto.pkl "
                "(miss_latent) yields Round18 eligible cohort (886)."
            ),
        },
    }
    (out_dir / "eval3_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def rebuild_round18_baseline(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Recompute metrics from Round18 Stage18E predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r18 = PROJECT_ROOT / cfg["paths"]["round18_root"] / "reports"
    arch = cfg["baseline"]["architecture_id"]
    fold_rows = []
    summary: Dict[str, Any] = {"architecture_id": arch, "targets": {}, "vs_gate": {}}
    gates = gate_table(cfg)

    for t in cfg["targets"]:
        key = t["key"]
        fold_path = r18 / f"round18e_tcga_{arch}_{key}_fold_predictions.csv"
        ens_path = r18 / f"round18e_tcga_{arch}_{key}_ensemble_predictions.csv"
        if not fold_path.is_file() or not ens_path.is_file():
            raise FileNotFoundError(f"Missing Round18 predictions for {key}: {fold_path}")

        fold_df = pd.read_csv(fold_path)
        ens_df = pd.read_csv(ens_path)
        fold_metrics = []
        for fold_id, gdf in fold_df.groupby("fold_id"):
            m = metrics_from_predictions(gdf, cfg)
            fold_metrics.append(
                {
                    "target_key": key,
                    "fold_id": int(fold_id),
                    "DrugMacro_AUC": m["DrugMacro_AUC"],
                    "DrugMacro_AUPRC": m["DrugMacro_AUPRC"],
                    "Global_AUC": m["Global_AUC"],
                    "Global_AUPRC": m["Global_AUPRC"],
                    "n_rows": m["n_rows"],
                    "n_valid_auc_drugs": m["n_valid_auc_drugs"],
                }
            )
        fold_rows.extend(fold_metrics)
        fm = pd.DataFrame(fold_metrics)
        ens_m = metrics_from_predictions(ens_df, cfg)
        avg_proxy = average_tcga_auc_proxy(ens_df)
        mean_auc = float(fm["DrugMacro_AUC"].mean())
        std_auc = float(fm["DrugMacro_AUC"].std(ddof=0))
        gate = gates[key]["gate_auroc"]
        summary["targets"][key] = {
            "fold_mean_DrugMacro_AUC": mean_auc,
            "fold_std_DrugMacro_AUC": std_auc,
            "fold_mean_DrugMacro_AUPRC": float(fm["DrugMacro_AUPRC"].mean()),
            "ensemble_DrugMacro_AUC": ens_m["DrugMacro_AUC"],
            "ensemble_DrugMacro_AUPRC": ens_m["DrugMacro_AUPRC"],
            "ensemble_Global_AUC": ens_m["Global_AUC"],
            "ensemble_Global_AUPRC": ens_m["Global_AUPRC"],
            "Average_TCGA_AUC_proxy": avg_proxy,
            "n_rows_eligible": ens_m["n_rows"],
            "headline_n_pairs": gates[key]["headline_n_pairs"],
            "gate_auroc": gate,
            "pass_gate_fold_mean": bool(mean_auc > gate),
            "delta_fold_mean_vs_gate": mean_auc - gate,
        }
        summary["vs_gate"][key] = {"pass": bool(mean_auc > gate), "delta": mean_auc - gate}

    pd.DataFrame(fold_rows).to_csv(out_dir / "baseline_fold_metrics.csv", index=False)
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Round 24 Stage 24A — Protocol Alignment Report",
        "",
        "## Cohort",
        "",
        "| Target | headline pairs | raw | eligible | dropped | miss_latent | miss_smiles |",
        "|--------|---------------:|----:|---------:|--------:|------------:|------------:|",
    ]
    cov = pd.read_csv(out_dir / "cohort_coverage.csv")
    for _, r in cov.iterrows():
        lines.append(
            f"| {r['target_key']} | {int(r['headline_n_pairs'])} | {int(r['n_raw'])} | "
            f"{int(r['n_eligible'])} | {int(r['n_dropped'])} | {int(r['n_miss_latent'])} | "
            f"{int(r['n_miss_smiles'])} |"
        )
    lines += [
        "",
        "## gdsc_intersect13 906 → 886",
        "",
        "Drop reason is exclusively `miss_latent` (patients absent from `tcga_latent_proto.pkl`).",
        "Round 24 formal gate uses the **eligible cohort** (same as Round 18 Stage 18E).",
        "Headline threshold table remains the product gate; paired comparisons must use eligible rows.",
        "",
        "## Baseline vs gate (Round18 pooled_mlp × own_plus_summary, 5-fold mean)",
        "",
        "| Target | fold-mean AUROC | gate | Δ | pass | eligible n | headline n |",
        "|--------|----------------:|-----:|--:|:----|----------:|-----------:|",
    ]
    for key, t in summary["targets"].items():
        lines.append(
            f"| {key} | {t['fold_mean_DrugMacro_AUC']:.4f} | {t['gate_auroc']:.4f} | "
            f"{t['delta_fold_mean_vs_gate']:+.4f} | {t['pass_gate_fold_mean']} | "
            f"{t['n_rows_eligible']} | {t['headline_n_pairs']} |"
        )
    lines += [
        "",
        "## Metric notes",
        "",
        "- Hard gate metric: 5-fold mean DrugMacro AUROC (support 10/2/2).",
        "- Ensemble AUROC is supporting only.",
        "- `Average_TCGA_AUC_proxy` = unfiltered per-drug AUC mean (historical style).",
        "",
    ]
    (out_dir / "protocol_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
