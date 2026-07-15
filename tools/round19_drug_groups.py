#!/usr/bin/env python3
"""Normalized drug identity table for Round 19E drug-held-out splits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_metrics import drug_is_valid
from tools.round18_eligible_data import _normalize_drug_key, load_smiles_lookup
from tools.round19_scaffold_groups import graph_canonical_smiles


def build_drug_group_table(
    development: pd.DataFrame,
    *,
    drug_smiles_path: str,
    drug_column: str = "DRUG_NAME",
    label_column: str = "Label",
) -> pd.DataFrame:
    """One row per display DRUG_NAME with normalized id + canonical SMILES."""
    if drug_column not in development.columns:
        raise KeyError(drug_column)
    smiles_lookup = load_smiles_lookup(drug_smiles_path)
    rows = []
    for drug, g in development.groupby(drug_column, dropna=False):
        drug_s = str(drug)
        norm = _normalize_drug_key(drug_s)
        raw_smiles = smiles_lookup.get(norm)
        if not raw_smiles:
            raise KeyError(f"No SMILES for drug {drug_s!r} (key={norm!r})")
        canon = graph_canonical_smiles(raw_smiles)
        if not canon:
            raise ValueError(f"Cannot canonicalize SMILES for {drug_s!r}")
        y = g[label_column].astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n = int(len(g))
        rows.append(
            {
                "DRUG_NAME": drug_s,
                "normalized_drug_id": norm,
                "canonical_smiles": canon,
                "n_rows": n,
                "n_modelids": int(g["ModelID"].nunique()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "auc_valid": bool(drug_is_valid(n, n_pos, n_neg)),
            }
        )
    table = pd.DataFrame(rows).sort_values("DRUG_NAME").reset_index(drop=True)
    # Hard: same normalized id must not map to multiple canonical SMILES
    for nid, g in table.groupby("normalized_drug_id"):
        smiles = set(g["canonical_smiles"])
        if len(smiles) > 1:
            raise AssertionError(
                f"normalized_drug_id={nid} maps to multiple canonical SMILES: {smiles}"
            )
        names = set(g["DRUG_NAME"])
        if len(names) > 1:
            raise AssertionError(
                f"normalized_drug_id={nid} has multiple DRUG_NAME aliases: {names}"
            )
    return table


def attach_normalized_drug_id(
    development: pd.DataFrame,
    drug_table: pd.DataFrame,
    *,
    drug_column: str = "DRUG_NAME",
) -> pd.DataFrame:
    m = drug_table.set_index("DRUG_NAME")["normalized_drug_id"].to_dict()
    out = development.copy()
    out["normalized_drug_id"] = out[drug_column].astype(str).map(m)
    if out["normalized_drug_id"].isna().any():
        missing = out.loc[out["normalized_drug_id"].isna(), drug_column].unique()[:10]
        raise AssertionError(f"Unmapped drugs: {list(missing)}")
    return out


def write_drug_group_table(root: Path, *, drug_smiles_path: str) -> Path:
    root = Path(root)
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(splits / "development_rows.csv")
    table = build_drug_group_table(development, drug_smiles_path=drug_smiles_path)
    path = splits / "round19e_drug_group_table.csv"
    table.to_csv(path, index=False)
    meta = {
        "n_drugs": int(len(table)),
        "n_auc_valid": int(table["auc_valid"].sum()),
        "path": str(path),
    }
    (splits / "round19e_drug_group_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument(
        "--drug-smiles-path",
        default="data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv",
    )
    args = parser.parse_args()
    path = write_drug_group_table(Path(args.root), drug_smiles_path=args.drug_smiles_path)
    print(json.dumps({"written": str(path)}, indent=2))


if __name__ == "__main__":
    main()
