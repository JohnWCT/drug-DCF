#!/usr/bin/env python3
"""Rename drug column and fill SMILES for TCGA ad intersect file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_map(df: pd.DataFrame, name_col: str, smiles_col: str) -> dict[str, str]:
    table = (
        df[[name_col, smiles_col]]
        .dropna(subset=[name_col])
        .assign(key=lambda d: d[name_col].astype(str).str.strip().str.lower())
    )
    return (
        table.dropna(subset=[smiles_col])
        .drop_duplicates(subset=["key"], keep="first")
        .set_index("key")[smiles_col]
        .to_dict()
    )


def main() -> None:
    ad_path = Path("/workspace/DAPL_git/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv")
    gdsc_path = Path("/workspace/DAPL_git/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv")
    map_path = Path("/workspace/DAPL_git/data/0_Drug_old_table/pdtc_gdsc_drug_mapping_all_addsmiles.csv")
    missing_out = Path(
        "/workspace/DAPL_git/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_missing_smiles_drugs.csv"
    )

    ad = pd.read_csv(ad_path)
    gdsc = pd.read_csv(gdsc_path)
    mapping = pd.read_csv(map_path)

    if "drug.name" in ad.columns:
        ad = ad.rename(columns={"drug.name": "drug_name"})
    if "drug_name" not in ad.columns:
        raise ValueError("Column 'drug_name' not found after rename step.")

    primary_map = build_map(gdsc, "DRUG_NAME", "SMILES")
    fallback_gdsc_map = build_map(mapping, "gdsc_name", "smiles")
    fallback_drug_map = build_map(mapping, "drug_name", "smiles")

    keys = ad["drug_name"].astype(str).str.strip().str.lower()

    ad["smiles"] = keys.map(primary_map)
    missing_before = int(ad["smiles"].isna().sum())

    mask = ad["smiles"].isna()
    ad.loc[mask, "smiles"] = keys[mask].map(fallback_gdsc_map)

    mask = ad["smiles"].isna()
    ad.loc[mask, "smiles"] = keys[mask].map(fallback_drug_map)
    missing_after = int(ad["smiles"].isna().sum())

    ad.to_csv(ad_path, index=False)

    missing_drugs = sorted(
        ad.loc[ad["smiles"].isna(), "drug_name"].astype(str).str.strip().str.lower().unique()
    )
    pd.DataFrame({"missing_drug_name_lower": missing_drugs}).to_csv(missing_out, index=False)

    print(f"updated_file: {ad_path}")
    print(f"missing_list_file: {missing_out}")
    print(f"rows: {len(ad)}")
    print(f"unique_drugs: {ad['drug_name'].astype(str).str.strip().str.lower().nunique()}")
    print(
        "unique_drugs_with_smiles: "
        f"{ad.loc[ad['smiles'].notna(), 'drug_name'].astype(str).str.strip().str.lower().nunique()}"
    )
    print(f"missing_smiles_rows_before_fallback: {missing_before}")
    print(f"missing_smiles_rows_after_fallback: {missing_after}")
    print(f"missing_unique_drugs_count: {len(missing_drugs)}")
    print(f"missing_unique_drugs: {missing_drugs}")


if __name__ == "__main__":
    main()
