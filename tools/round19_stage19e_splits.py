#!/usr/bin/env python3
"""Build and QC Round 19E drug / scaffold / cancer-type held-out 5CV splits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_metrics import drug_is_valid
from tools.round18_eligible_data import load_smiles_lookup
from tools.round19_cancer_type_groups import (
    attach_cancer_type,
    build_modelid_cancer_type_map,
    write_cancer_type_group_table,
)
from tools.round19_cv_splits import build_heldout_assignments
from tools.round19_drug_groups import (
    attach_normalized_drug_id,
    build_drug_group_table,
    write_drug_group_table,
)
from tools.round19_scaffold_groups import build_scaffold_map, graph_canonical_smiles

SHIFT_SEEDS = {
    "drug_heldout": 19051,
    "scaffold_heldout": 19061,
    "cancer_type_heldout": 19071,
}

ASSIGNMENT_NAMES = {
    "drug_heldout": "round19e_drug_heldout_5cv.csv",
    "scaffold_heldout": "round19e_scaffold_heldout_5cv.csv",
    "cancer_type_heldout": "round19e_cancer_type_heldout_5cv.csv",
}


def _count_valid_drugs(val_df: pd.DataFrame) -> int:
    n = 0
    for _, g in val_df.groupby("DRUG_NAME"):
        y = g["Label"].astype(int)
        if drug_is_valid(len(g), int((y == 1).sum()), int((y == 0).sum())):
            n += 1
    return n


def _base_qc(
    assignments: pd.DataFrame,
    *,
    development: pd.DataFrame,
    internal_test: pd.DataFrame,
    group_column: str,
    n_folds: int,
    min_valid_drugs: int = 3,
    strategy: str,
) -> List[dict]:
    dev_rows = set(development["_row_id"].astype(int))
    it_rows = set(internal_test["_row_id"].astype(int))
    assign_rows = set(assignments["_row_id"].astype(int))
    if assign_rows != dev_rows:
        raise AssertionError(f"{strategy}: assignment rows != development")
    if assign_rows & it_rows:
        raise AssertionError(f"{strategy}: internal-test rows leaked")
    if set(assignments["ModelID"].astype(str)) & set(internal_test["ModelID"].astype(str)):
        raise AssertionError(f"{strategy}: internal-test ModelIDs leaked")

    val = assignments[assignments["split_role"] == "val"]
    if val["_row_id"].duplicated().any():
        raise AssertionError(f"{strategy}: duplicate val rows")
    if set(val["_row_id"].astype(int)) != dev_rows:
        raise AssertionError(f"{strategy}: val coverage incomplete")

    qc = []
    for fold_id in range(n_folds):
        f = assignments[assignments["fold_id"].astype(int) == fold_id]
        train = f[f["split_role"] == "train"]
        vald = f[f["split_role"] == "val"]
        tg = set(train[group_column].astype(str))
        vg = set(vald[group_column].astype(str))
        if tg & vg:
            raise AssertionError(f"{strategy} fold={fold_id}: group overlap on {group_column}")
        val_full = development.merge(vald[["_row_id"]], on="_row_id", how="inner")
        n_valid = _count_valid_drugs(val_full)
        if n_valid < min_valid_drugs:
            raise AssertionError(
                f"{strategy} fold={fold_id}: valid DrugMacro drugs={n_valid} < {min_valid_drugs}"
            )
        qc.append(
            {
                "shift_strategy": strategy,
                "fold_id": int(fold_id),
                "n_train_rows": int(len(train)),
                "n_val_rows": int(len(vald)),
                "n_train_groups": int(len(tg)),
                "n_val_groups": int(len(vg)),
                "positive_rate": float(vald["Label"].astype(int).mean()) if len(vald) else 0.0,
                "valid_drugmacro_drugs": int(n_valid),
            }
        )
    return qc


def _write_scaffold_table(root: Path, development: pd.DataFrame, drug_smiles_path: str) -> pd.DataFrame:
    smiles_lookup = load_smiles_lookup(drug_smiles_path)
    drug_to_smiles = {}
    for d in development["DRUG_NAME"].astype(str).unique():
        key = d.strip().lower()
        if key not in smiles_lookup:
            raise KeyError(d)
        drug_to_smiles[d] = smiles_lookup[key]
    smap = build_scaffold_map(drug_to_smiles)
    rows = []
    for drug, sid in smap.items():
        g = development[development["DRUG_NAME"].astype(str) == drug]
        y = g["Label"].astype(int)
        canon = graph_canonical_smiles(drug_to_smiles[drug])
        rows.append(
            {
                "DRUG_NAME": drug,
                "scaffold_id": sid,
                "canonical_smiles": canon,
                "n_rows": int(len(g)),
                "n_modelids": int(g["ModelID"].nunique()),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "is_acyclic_fallback": sid.startswith("ACYCLIC:"),
            }
        )
    table = pd.DataFrame(rows)
    # size warning
    total = int(len(development))
    for sid, g in table.groupby("scaffold_id"):
        frac = float(g["n_rows"].sum()) / max(total, 1)
        if frac > 0.20:
            print(
                f"[WARN] scaffold {sid} covers {frac:.1%} of development rows",
                file=sys.stderr,
            )
    path = root / "splits" / "round19e_scaffold_group_table.csv"
    table.to_csv(path, index=False)
    return table


def build_drug_heldout(
    root: Path,
    *,
    drug_smiles_path: str,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, List[dict]]:
    splits = root / "splits"
    development = pd.read_csv(splits / "development_rows.csv")
    internal_test = pd.read_csv(splits / "internal_test_split.csv")
    write_drug_group_table(root, drug_smiles_path=drug_smiles_path)
    drug_table = pd.read_csv(splits / "round19e_drug_group_table.csv")
    tagged = attach_normalized_drug_id(development, drug_table)
    assign = build_heldout_assignments(
        tagged,
        group_column="normalized_drug_id",
        n_splits=n_folds,
        split_seed=SHIFT_SEEDS["drug_heldout"],
        cv_name="round19e_drug_heldout_5fold",
    )
    # Enrich
    cmap = tagged.set_index("_row_id")
    for col in ("DRUG_NAME", "normalized_drug_id", "ModelID", "Label"):
        assign[col] = assign["_row_id"].map(cmap[col])
    assign["partition"] = assign["split_role"]
    assign["split_seed"] = SHIFT_SEEDS["drug_heldout"]
    assign["shift_strategy"] = "drug_heldout"
    # drug overlap already covered by group QC; also assert DRUG_NAME
    for fold_id in range(n_folds):
        f = assign[assign.fold_id == fold_id]
        tr = set(f.loc[f.split_role == "train", "DRUG_NAME"].astype(str))
        va = set(f.loc[f.split_role == "val", "DRUG_NAME"].astype(str))
        if tr & va:
            raise AssertionError(f"drug_heldout fold={fold_id}: DRUG_NAME overlap")
    qc = _base_qc(
        assign,
        development=development,
        internal_test=internal_test,
        group_column="normalized_drug_id",
        n_folds=n_folds,
        strategy="drug_heldout",
    )
    return assign, qc


def build_scaffold_heldout(
    root: Path,
    *,
    drug_smiles_path: str,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, List[dict]]:
    splits = root / "splits"
    development = pd.read_csv(splits / "development_rows.csv")
    internal_test = pd.read_csv(splits / "internal_test_split.csv")
    scaffold_table = _write_scaffold_table(root, development, drug_smiles_path)
    smap = scaffold_table.set_index("DRUG_NAME")["scaffold_id"].to_dict()
    tagged = development.copy()
    tagged["scaffold_id"] = tagged["DRUG_NAME"].astype(str).map(smap)
    tagged["canonical_smiles"] = tagged["DRUG_NAME"].astype(str).map(
        scaffold_table.set_index("DRUG_NAME")["canonical_smiles"].to_dict()
    )
    assign = build_heldout_assignments(
        tagged,
        group_column="scaffold_id",
        n_splits=n_folds,
        split_seed=SHIFT_SEEDS["scaffold_heldout"],
        cv_name="round19e_scaffold_heldout_5fold",
    )
    cmap = tagged.set_index("_row_id")
    for col in ("DRUG_NAME", "scaffold_id", "canonical_smiles", "ModelID", "Label"):
        assign[col] = assign["_row_id"].map(cmap[col])
    assign["partition"] = assign["split_role"]
    assign["split_seed"] = SHIFT_SEEDS["scaffold_heldout"]
    assign["shift_strategy"] = "scaffold_heldout"
    # Extra: drug / smiles not across folds
    for fold_id in range(n_folds):
        f = assign[assign.fold_id == fold_id]
        for col in ("scaffold_id", "DRUG_NAME", "canonical_smiles"):
            tr = set(f.loc[f.split_role == "train", col].astype(str))
            va = set(f.loc[f.split_role == "val", col].astype(str))
            if tr & va:
                raise AssertionError(f"scaffold_heldout fold={fold_id}: {col} overlap")
    qc = _base_qc(
        assign,
        development=development,
        internal_test=internal_test,
        group_column="scaffold_id",
        n_folds=n_folds,
        strategy="scaffold_heldout",
    )
    for q in qc:
        q["n_acyclic_fallback"] = int(scaffold_table["is_acyclic_fallback"].sum())
    return assign, qc


def build_cancer_type_heldout(
    root: Path,
    *,
    n_folds: int = 5,
    allow_reduce_to_3: bool = True,
) -> Tuple[pd.DataFrame, List[dict], int]:
    splits = root / "splits"
    development = pd.read_csv(splits / "development_rows.csv")
    internal_test = pd.read_csv(splits / "internal_test_split.csv")
    write_cancer_type_group_table(root)
    mapping = pd.read_csv(splits / "round19e_modelid_cancer_type_map.csv")
    tagged = attach_cancer_type(development, mapping)
    used_folds = n_folds
    reason = None
    try:
        assign = build_heldout_assignments(
            tagged,
            group_column="cancer_type",
            n_splits=used_folds,
            split_seed=SHIFT_SEEDS["cancer_type_heldout"],
            cv_name=f"round19e_cancer_type_heldout_{used_folds}fold",
        )
        cmap = tagged.set_index("_row_id")
        for col in ("DRUG_NAME", "cancer_type", "ModelID", "Label"):
            assign[col] = assign["_row_id"].map(cmap[col])
        # Pre-check: >=2 cancer types per val fold
        for fold_id in range(used_folds):
            f = assign[(assign.fold_id == fold_id) & (assign.split_role == "val")]
            if f["cancer_type"].nunique() < 2:
                raise AssertionError(f"cancer fold={fold_id}: <2 val cancer types")
        qc = _base_qc(
            assign,
            development=development,
            internal_test=internal_test,
            group_column="cancer_type",
            n_folds=used_folds,
            strategy="cancer_type_heldout",
        )
        # ModelID overlap
        for fold_id in range(used_folds):
            f = assign[assign.fold_id == fold_id]
            tr = set(f.loc[f.split_role == "train", "ModelID"].astype(str))
            va = set(f.loc[f.split_role == "val", "ModelID"].astype(str))
            if tr & va:
                raise AssertionError(f"cancer fold={fold_id}: ModelID overlap")
    except Exception as exc:  # noqa: BLE001
        if not allow_reduce_to_3 or n_folds <= 3:
            raise
        reason = f"5-fold QC failed ({exc}); reducing to 3-fold before any training"
        used_folds = 3
        assign = build_heldout_assignments(
            tagged,
            group_column="cancer_type",
            n_splits=used_folds,
            split_seed=SHIFT_SEEDS["cancer_type_heldout"],
            cv_name=f"round19e_cancer_type_heldout_{used_folds}fold",
        )
        cmap = tagged.set_index("_row_id")
        for col in ("DRUG_NAME", "cancer_type", "ModelID", "Label"):
            assign[col] = assign["_row_id"].map(cmap[col])
        for fold_id in range(used_folds):
            f = assign[(assign.fold_id == fold_id) & (assign.split_role == "val")]
            if f["cancer_type"].nunique() < 2:
                raise AssertionError(f"cancer 3-fold fold={fold_id}: <2 val cancer types")
        qc = _base_qc(
            assign,
            development=development,
            internal_test=internal_test,
            group_column="cancer_type",
            n_folds=used_folds,
            strategy="cancer_type_heldout",
        )
        for fold_id in range(used_folds):
            f = assign[assign.fold_id == fold_id]
            tr = set(f.loc[f.split_role == "train", "ModelID"].astype(str))
            va = set(f.loc[f.split_role == "val", "ModelID"].astype(str))
            if tr & va:
                raise AssertionError(f"cancer 3-fold fold={fold_id}: ModelID overlap")
        for q in qc:
            q["fold_reduction_reason"] = reason

    assign["partition"] = assign["split_role"]
    assign["split_seed"] = SHIFT_SEEDS["cancer_type_heldout"]
    assign["shift_strategy"] = "cancer_type_heldout"
    for q in qc:
        q["n_folds_used"] = used_folds
        if reason:
            q["fold_reduction_reason"] = reason
    return assign, qc, used_folds


def build_all_round19e_splits(
    root: Path,
    *,
    drug_smiles_path: str,
) -> Dict[str, str]:
    root = Path(root)
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    qc_all: List[dict] = []

    drug_asg, drug_qc = build_drug_heldout(root, drug_smiles_path=drug_smiles_path)
    path = splits / ASSIGNMENT_NAMES["drug_heldout"]
    drug_asg.to_csv(path, index=False)
    written["drug_heldout"] = str(path)
    qc_all.extend(drug_qc)

    scaf_asg, scaf_qc = build_scaffold_heldout(root, drug_smiles_path=drug_smiles_path)
    path = splits / ASSIGNMENT_NAMES["scaffold_heldout"]
    scaf_asg.to_csv(path, index=False)
    written["scaffold_heldout"] = str(path)
    qc_all.extend(scaf_qc)

    can_asg, can_qc, n_folds_cancer = build_cancer_type_heldout(root)
    # Always write under the canonical 5cv filename; metadata records actual folds
    path = splits / ASSIGNMENT_NAMES["cancer_type_heldout"]
    can_asg.to_csv(path, index=False)
    written["cancer_type_heldout"] = str(path)
    qc_all.extend(can_qc)

    pd.DataFrame(qc_all).to_csv(splits / "round19e_split_qc.csv", index=False)
    meta = {
        "shift_seeds": SHIFT_SEEDS,
        "paths": written,
        "cancer_type_n_folds": int(n_folds_cancer),
        "internal_test_regenerated": False,
    }
    (splits / "round19e_split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument(
        "--drug-smiles-path",
        default="data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv",
    )
    args = parser.parse_args()
    written = build_all_round19e_splits(Path(args.root), drug_smiles_path=args.drug_smiles_path)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
