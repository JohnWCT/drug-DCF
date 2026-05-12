#!/usr/bin/env python3
"""Standardize drug_name columns across key CSV files using GDSC mapping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


GDSC_PATH = Path("/workspace/DAPL_git/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv")
PDX_PATH = Path("/workspace/DAPL_git/data/PDX/PDX_drug_response_from_DAPL.csv")
GDSC2_PATH = Path("/workspace/DAPL_git/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv")
TCGA_PATH = Path("/workspace/DAPL_git/data/TCGA/TCGA_drug_response_from_DAPL.csv")
MISSING_SMILES_PATH = Path(
    "/workspace/DAPL_git/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_missing_smiles_drugs.csv"
)


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_gdsc_maps(gdsc: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    names = gdsc["DRUG_NAME"].dropna().astype(str).str.strip()
    lower_map = {name.lower(): name.lower() for name in names}
    norm_map = {norm_key(name): name.lower() for name in names}
    return lower_map, norm_map


def standardize_series(
    series: pd.Series, lower_map: dict[str, str], norm_map: dict[str, str]
) -> tuple[pd.Series, list[str]]:
    out = []
    unresolved: list[str] = []
    for value in series.astype(str):
        key_raw = value.strip()
        key_low = key_raw.lower()
        if key_low in lower_map:
            out.append(lower_map[key_low])
            continue
        key_norm = norm_key(key_raw)
        if key_norm in norm_map:
            out.append(norm_map[key_norm])
            continue
        out.append(key_low)
        unresolved.append(key_raw)
    return pd.Series(out, index=series.index), sorted(set(unresolved))


def ensure_drug_name_column(df: pd.DataFrame, current_col: str) -> pd.DataFrame:
    if current_col == "drug_name":
        return df
    if current_col in df.columns:
        return df.rename(columns={current_col: "drug_name"})
    raise ValueError(f"Column '{current_col}' not found.")


def process_file(
    path: Path,
    input_col: str,
    lower_map: dict[str, str],
    norm_map: dict[str, str],
) -> list[str]:
    df = pd.read_csv(path)
    df = ensure_drug_name_column(df, input_col)
    standardized, unresolved = standardize_series(df["drug_name"], lower_map, norm_map)
    df["drug_name"] = standardized
    df.to_csv(path, index=False)
    return unresolved


def process_missing_file(
    path: Path,
    lower_map: dict[str, str],
    norm_map: dict[str, str],
) -> list[str]:
    df = pd.read_csv(path)
    if "drug_name" not in df.columns:
        if "missing_drug_name_lower" in df.columns:
            df = df.rename(columns={"missing_drug_name_lower": "drug_name"})
        else:
            df["drug_name"] = []
    if len(df) == 0:
        df = df[["drug_name"]]
        df.to_csv(path, index=False)
        return []
    standardized, unresolved = standardize_series(df["drug_name"], lower_map, norm_map)
    df["drug_name"] = standardized
    df = df[["drug_name"]]
    df.to_csv(path, index=False)
    return unresolved


def summarize(path: Path, unresolved: Iterable[str]) -> None:
    df = pd.read_csv(path)
    unresolved_list = sorted(set(unresolved))
    print(f"[done] {path}")
    print(f"  rows={len(df)}")
    if "drug_name" in df.columns:
        print(f"  unique_drug_name={df['drug_name'].astype(str).nunique()}")
    print(f"  unresolved={len(unresolved_list)}")
    if unresolved_list:
        print(f"  unresolved_examples={unresolved_list[:10]}")


def main() -> None:
    gdsc = pd.read_csv(GDSC_PATH)
    lower_map, norm_map = build_gdsc_maps(gdsc)

    # Add unified drug_name column in GDSC table.
    gdsc["drug_name"] = gdsc["DRUG_NAME"].astype(str).str.strip().str.lower()
    gdsc.to_csv(GDSC_PATH, index=False)
    summarize(GDSC_PATH, [])

    unresolved_pdx = process_file(PDX_PATH, "drug_name", lower_map, norm_map)
    summarize(PDX_PATH, unresolved_pdx)

    unresolved_gdsc2 = process_file(GDSC2_PATH, "drug_name", lower_map, norm_map)
    summarize(GDSC2_PATH, unresolved_gdsc2)

    unresolved_tcga = process_file(TCGA_PATH, "drug_name", lower_map, norm_map)
    summarize(TCGA_PATH, unresolved_tcga)

    unresolved_missing = process_missing_file(MISSING_SMILES_PATH, lower_map, norm_map)
    summarize(MISSING_SMILES_PATH, unresolved_missing)


if __name__ == "__main__":
    main()
