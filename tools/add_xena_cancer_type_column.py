"""Populate cancer_type in TCGA/CCLE sample info CSVs (full values, including 'na')."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.tcga_cancer_type_maps import build_tcga_name_to_cancer_type_map, map_primary_disease_to_cancer_type


def add_tcga_cancer_type_column(csv_path: str, inplace: bool = True) -> pd.DataFrame:
    name_to_type = build_tcga_name_to_cancer_type_map()
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    if "_primary_disease" not in df.columns:
        raise ValueError(f"Missing _primary_disease column in {csv_path}")
    df["cancer_type"] = df["_primary_disease"].astype(str).map(
        lambda x: map_primary_disease_to_cancer_type(x, name_to_type)
    )
    na_count = int((df["cancer_type"] == "na").sum())
    if inplace:
        df.to_csv(csv_path)
        print(f"Updated {csv_path} ({len(df)} rows); cancer_type na={na_count}")
    return df


def add_ccle_cancer_type_column(
    csv_path: str = os.path.join("data", "ccle_sample_info_df.csv"),
    inplace: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    if "primary_disease" not in df.columns:
        raise ValueError(f"Missing primary_disease column in {csv_path}")
    df["cancer_type"] = df["primary_disease"].astype(str).str.strip()
    df.loc[df["primary_disease"].isna(), "cancer_type"] = "na"
    if inplace:
        df.to_csv(csv_path)
        print(f"Updated {csv_path} ({len(df)} rows); columns include cancer_type")
    return df


def main():
    parser = argparse.ArgumentParser(description="Add full cancer_type to TCGA/CCLE sample info CSVs")
    parser.add_argument("--tcga-csv", default=os.path.join("data", "TCGA", "xena_sample_info_df.csv"))
    parser.add_argument("--ccle-csv", default=os.path.join("data", "ccle_sample_info_df.csv"))
    parser.add_argument("--tcga-only", action="store_true")
    parser.add_argument("--ccle-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    inplace = not args.dry_run
    if not args.ccle_only:
        add_tcga_cancer_type_column(args.tcga_csv, inplace=inplace)
    if not args.tcga_only:
        add_ccle_cancer_type_column(args.ccle_csv, inplace=inplace)


if __name__ == "__main__":
    main()
