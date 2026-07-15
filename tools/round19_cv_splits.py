"""Round 19 CV split helpers (ModelID / drug / scaffold / cancer-type)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from tools.round18_cv_metrics import drug_is_valid
from tools.round18_cv_splits import build_grouped_cv_assignments
from tools.round19_scaffold_groups import build_scaffold_map, murcko_scaffold_id


def link_or_reuse_round18_splits(round18_outdir: str, round19_outdir: str) -> Dict[str, str]:
    src = Path(round18_outdir) / "splits"
    dst = Path(round19_outdir) / "splits"
    dst.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for name in [
        "screening_3fold_assignments.csv",
        "formal_5fold_assignments.csv",
        "internal_test_split.csv",
        "development_rows.csv",
        "split_metadata.json",
    ]:
        s = src / name
        d = dst / name
        if not s.is_file():
            raise FileNotFoundError(s)
        if d.exists() or d.is_symlink():
            if d.is_symlink() or d.is_file():
                d.unlink()
            else:
                shutil.rmtree(d)
        try:
            d.symlink_to(s.resolve())
        except OSError:
            shutil.copy2(s, d)
        mapping[name] = str(d)
    return mapping


def link_or_reuse_round18_eligible(round18_outdir: str, round19_outdir: str) -> str:
    src = Path(round18_outdir) / "data" / "round18_eligible_response.csv"
    dst_dir = Path(round19_outdir) / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "round19_eligible_response.csv"
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def build_heldout_assignments(
    eligible_df: pd.DataFrame,
    *,
    group_column: str,
    n_splits: int = 5,
    split_seed: int = 52,
    cv_name: str = "drug_heldout_5fold",
    label_column: str = "Label",
) -> pd.DataFrame:
    """Generic stratified-group held-out CV over an arbitrary group column."""
    return build_grouped_cv_assignments(
        eligible_df,
        n_splits=n_splits,
        group_column=group_column,
        label_column=label_column,
        split_seed=split_seed,
        cv_name=cv_name,
    )


def attach_scaffold_ids(
    eligible_df: pd.DataFrame,
    drug_to_smiles: Dict[str, str],
    *,
    drug_column: str = "DRUG_NAME",
) -> pd.DataFrame:
    df = eligible_df.copy()
    smap = build_scaffold_map(drug_to_smiles)
    df["scaffold_id"] = df[drug_column].astype(str).map(
        lambda d: smap[d] if d in smap else murcko_scaffold_id(drug_to_smiles[d])
    )
    return df


def _count_valid_drugmacro_drugs(
    val_df: pd.DataFrame,
    *,
    drug_column: str = "DRUG_NAME",
    label_column: str = "Label",
) -> int:
    n_valid = 0
    for _, g in val_df.groupby(drug_column):
        y = g[label_column].astype(int)
        if drug_is_valid(len(g), int((y == 1).sum()), int((y == 0).sum())):
            n_valid += 1
    return n_valid


def validate_round19d_assignments(
    assignments: pd.DataFrame,
    *,
    development: pd.DataFrame,
    internal_test: pd.DataFrame,
    split_seed: int,
    n_folds: int = 5,
    min_valid_drugs: int = 3,
) -> Dict:
    """Hard QC for a Round 19D 5CV assignment table."""
    dev_rows = set(development["_row_id"].astype(int))
    it_rows = set(internal_test["_row_id"].astype(int))
    it_mids = set(internal_test["ModelID"].astype(str))
    assign_rows = set(assignments["_row_id"].astype(int))
    if assign_rows != dev_rows:
        raise AssertionError(
            f"seed={split_seed}: assignment rows != development "
            f"(missing={len(dev_rows - assign_rows)} extra={len(assign_rows - dev_rows)})"
        )
    if assign_rows & it_rows:
        raise AssertionError(f"seed={split_seed}: internal-test rows leaked into assignments")
    if set(assignments["ModelID"].astype(str)) & it_mids:
        raise AssertionError(f"seed={split_seed}: internal-test ModelIDs in assignments")

    fold_ids = set(assignments["fold_id"].astype(int))
    if fold_ids != set(range(n_folds)):
        raise AssertionError(f"seed={split_seed}: fold_ids={fold_ids}")

    # Each development row appears as val exactly once
    val = assignments[assignments["split_role"] == "val"]
    if val["_row_id"].duplicated().any():
        raise AssertionError(f"seed={split_seed}: duplicate val _row_id")
    if set(val["_row_id"].astype(int)) != dev_rows:
        raise AssertionError(f"seed={split_seed}: val coverage incomplete")

    qc_folds = []
    for fold_id in range(n_folds):
        f = assignments[assignments["fold_id"].astype(int) == fold_id]
        train = f[f["split_role"] == "train"]
        vald = f[f["split_role"] == "val"]
        train_m = set(train["ModelID"].astype(str))
        val_m = set(vald["ModelID"].astype(str))
        if train_m & val_m:
            raise AssertionError(f"seed={split_seed} fold={fold_id}: ModelID overlap")
        # enrich with drug/label from development for DrugMacro support
        val_full = development.merge(vald[["_row_id"]], on="_row_id", how="inner")
        n_valid = _count_valid_drugmacro_drugs(val_full)
        if n_valid < min_valid_drugs:
            raise AssertionError(
                f"seed={split_seed} fold={fold_id}: valid DrugMacro drugs={n_valid} < {min_valid_drugs}"
            )
        qc_folds.append(
            {
                "split_seed": int(split_seed),
                "fold_id": int(fold_id),
                "n_train_rows": int(len(train)),
                "n_val_rows": int(len(vald)),
                "n_train_modelids": int(len(train_m)),
                "n_val_modelids": int(len(val_m)),
                "positive_rate": float(vald["Label"].astype(int).mean()) if len(vald) else 0.0,
                "valid_drugmacro_drugs": int(n_valid),
            }
        )
    # val ModelIDs across folds are mutually exclusive
    val_sets = [
        set(assignments[(assignments.fold_id == f) & (assignments.split_role == "val")]["ModelID"].astype(str))
        for f in range(n_folds)
    ]
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            if val_sets[i] & val_sets[j]:
                raise AssertionError(f"seed={split_seed}: val ModelID overlap folds {i}/{j}")
    return {"split_seed": int(split_seed), "folds": qc_folds}


def build_round19d_splits(
    root: Path,
    *,
    split_seeds: Sequence[int] = (52, 62, 72),
    n_folds: int = 5,
) -> Dict[str, str]:
    """
    Build confirmation 5CV assignments on the locked development 90%.
    Internal test is never regenerated.
    """
    root = Path(root)
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(splits / "development_rows.csv")
    internal_test = pd.read_csv(splits / "internal_test_split.csv")
    if "_row_id" not in development.columns:
        raise KeyError("development_rows.csv missing _row_id")
    # Ensure Label/ModelID/DRUG_NAME present
    for col in ("ModelID", "Label", "DRUG_NAME"):
        if col not in development.columns:
            raise KeyError(f"development_rows.csv missing {col}")

    written: Dict[str, str] = {}
    qc_all: List[dict] = []
    for seed in split_seeds:
        assign = build_grouped_cv_assignments(
            development,
            n_splits=int(n_folds),
            group_column="ModelID",
            label_column="Label",
            split_seed=int(seed),
            cv_name=f"round19d_seed{seed}_5cv",
        )
        # Enrich with DRUG_NAME for QC/reporting
        drug_map = development.set_index("_row_id")["DRUG_NAME"].to_dict()
        assign["DRUG_NAME"] = assign["_row_id"].map(drug_map)
        assign["split_seed"] = int(seed)
        assign = assign.rename(columns={"split_role": "partition"}) if False else assign
        # Keep split_role for pipeline compatibility; also expose partition alias
        assign["partition"] = assign["split_role"]
        qc = validate_round19d_assignments(
            assign,
            development=development,
            internal_test=internal_test,
            split_seed=int(seed),
            n_folds=int(n_folds),
        )
        qc_all.extend(qc["folds"])
        out = splits / f"round19d_seed{seed}_5cv_assignments.csv"
        # Standard columns for pipeline subset_by_assignment (uses split_role)
        cols = [
            "cv_name",
            "fold_id",
            "split_role",
            "partition",
            "_row_id",
            "ModelID",
            "DRUG_NAME",
            "Label",
            "split_seed",
        ]
        assign[cols].to_csv(out, index=False)
        written[str(seed)] = str(out)
    (splits / "round19d_split_qc.csv").write_text(
        pd.DataFrame(qc_all).to_csv(index=False), encoding="utf-8"
    )
    (splits / "round19d_split_metadata.json").write_text(
        json.dumps(
            {
                "split_seeds": list(map(int, split_seeds)),
                "n_folds": int(n_folds),
                "internal_test_regenerated": False,
                "group_column": "ModelID",
                "label_column": "Label",
                "paths": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return written
