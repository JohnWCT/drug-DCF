#!/usr/bin/env python3
"""Prepare NoHoldout / AACDR training tables + formal 5-fold assignments for Round24 ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round18_cv_splits import build_grouped_cv_assignments
from tools.round18_eligible_data import (
    _normalize_drug_key,
    build_round18_eligible_response,
    load_omics_latent_dict,
    load_smiles_lookup,
    try_build_graph,
    validate_feature_metadata,
)

FEATURE_OWN = ROOT / "result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary"
SMILES = ROOT / "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"
R18 = ROOT / "result/optimization_runs/round18_architecture"
OUT = ROOT / "result/optimization_runs/round24_train_source_ablation"


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def prepare_ctrl() -> Dict[str, Any]:
    """Symlink/copy Round18 development + folds as Ctrl (no retrain needed for metrics)."""
    arm = OUT / "Ctrl"
    arm.mkdir(parents=True, exist_ok=True)
    (arm / "splits").mkdir(exist_ok=True)
    for name in ("development_rows.csv", "formal_5fold_assignments.csv"):
        src = R18 / "splits" / name
        dst = arm / "splits" / name
        if dst.is_file() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    summary = {
        "arm": "Ctrl",
        "source": "round18_development",
        "n_development": int(sum(1 for _ in open(R18 / "splits" / "development_rows.csv")) - 1),
        "holdout": "kept_internal_test_out_of_cv",
        "feature_dir": str(FEATURE_OWN),
    }
    _write_json(arm / "arm_summary.json", summary)
    return summary


def prepare_noholdout(*, split_seed: int = 42) -> Dict[str, Any]:
    arm = OUT / "NoHoldout"
    splits = arm / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(R18 / "splits" / "development_rows.csv")
    it = pd.read_csv(R18 / "splits" / "internal_test_split.csv")
    # Align columns
    cols = [c for c in dev.columns if c in it.columns]
    full = pd.concat([dev[cols], it[cols]], ignore_index=True)
    full = full.reset_index(drop=True)
    full["_row_id"] = np.arange(len(full), dtype=int)
    if "DRUG_NAME" not in full.columns:
        drug_col = "mapped_name" if "mapped_name" in full.columns else "drug_name"
        full["DRUG_NAME"] = full[drug_col].astype(str)
    full.to_csv(splits / "development_rows.csv", index=False)
    assigns = build_grouped_cv_assignments(
        full,
        n_splits=5,
        group_column="ModelID",
        label_column="Label",
        split_seed=split_seed,
        cv_name="formal_5fold",
    )
    assigns.to_csv(splits / "formal_5fold_assignments.csv", index=False)
    summary = {
        "arm": "NoHoldout",
        "source": "GDSC2_development_union_internal_test",
        "n_development": int(len(full)),
        "n_from_dev": int(len(dev)),
        "n_from_internal_test": int(len(it)),
        "holdout": "disabled_full_5fold_cv",
        "split_seed": split_seed,
        "feature_dir": str(FEATURE_OWN),
        "n_model_ids": int(full["ModelID"].nunique()),
        "n_drugs": int(full["DRUG_NAME"].nunique()),
        "positive_rate": float(full["Label"].mean()),
    }
    _write_json(arm / "arm_summary.json", summary)
    return summary


def prepare_aacdr(*, split_seed: int = 42) -> Dict[str, Any]:
    """Build eligible AACDR rows via Round18 eligibility filters, then 5-fold CV (no holdout)."""
    arm = OUT / "AACDR"
    arm.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data/AACDR_gdsc_binary_response_151_long_for_dapl_exclude_tcga_only.csv"
    raw = pd.read_csv(raw_path)
    # Normalize schema toward Round18 eligible builder
    work = raw.copy()
    work["ModelID"] = work["Sample_ID"].astype(str)
    work["drug_name"] = work["drug_name"].astype(str)
    work["Label"] = work["Label"].astype(int)
    work["DATASET"] = work.get("DATASET", "AACDR_GDSC151")
    staged = arm / "aacdr_response_staged.csv"
    work.to_csv(staged, index=False)

    elig = build_round18_eligible_response(
        str(staged),
        feature_dir=str(FEATURE_OWN),
        drug_smiles_path=str(SMILES),
        outdir=str(arm),
        group_column="ModelID",
        label_column="Label",
        drug_column_candidates=("drug_name", "DRUG_NAME"),
    )
    eligible_path = Path(elig["paths"]["eligible"])
    eligible = pd.read_csv(eligible_path)
    # Use as development (no 10% holdout)
    splits = arm / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    eligible = eligible.reset_index(drop=True)
    eligible["_row_id"] = np.arange(len(eligible), dtype=int)
    if "DRUG_NAME" not in eligible.columns:
        eligible["DRUG_NAME"] = eligible["drug_name"].astype(str)
    eligible.to_csv(splits / "development_rows.csv", index=False)
    assigns = build_grouped_cv_assignments(
        eligible,
        n_splits=5,
        group_column="ModelID",
        label_column="Label",
        split_seed=split_seed,
        cv_name="formal_5fold",
    )
    assigns.to_csv(splits / "formal_5fold_assignments.csv", index=False)
    summary = {
        "arm": "AACDR",
        "source": "AACDR_gdsc_binary_response_151_long_for_dapl_exclude_tcga_only",
        "n_raw": int(len(raw)),
        "n_eligible": int(len(eligible)),
        "n_dropped": int(len(raw) - len(eligible)),
        "eligibility": elig,
        "holdout": "disabled_full_5fold_cv",
        "split_seed": split_seed,
        "feature_dir": str(FEATURE_OWN),
        "n_model_ids": int(eligible["ModelID"].nunique()),
        "n_drugs": int(eligible["DRUG_NAME"].nunique()),
        "positive_rate": float(eligible["Label"].mean()),
    }
    _write_json(arm / "arm_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=["Ctrl", "NoHoldout", "AACDR"])
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for arm in args.arms:
        if arm == "Ctrl":
            reports[arm] = prepare_ctrl()
        elif arm == "NoHoldout":
            reports[arm] = prepare_noholdout(split_seed=args.split_seed)
        elif arm == "AACDR":
            reports[arm] = prepare_aacdr(split_seed=args.split_seed)
        else:
            raise SystemExit(f"Unknown arm {arm}")
        print(json.dumps(reports[arm], indent=2, default=str)[:1500], flush=True)
    _write_json(OUT / "prepare_summary.json", reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
