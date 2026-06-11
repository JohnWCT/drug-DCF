#!/usr/bin/env python3
"""Split TCGA intersect_pretrain CSV by GDSC drug overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# GDSC2 mapped_name ∩ TCGA (lowercase)
GDSC_INTERSECT_13 = {
    "5-fluorouracil",
    "bicalutamide",
    "bleomycin",
    "cisplatin",
    "docetaxel",
    "gemcitabine",
    "methotrexate",
    "paclitaxel",
    "sorafenib",
    "tamoxifen",
    "temozolomide",
    "vinblastine",
    "vinorelbine",
}
TCGA_ONLY_3 = {"doxorubicin", "etoposide", "pemetrexed"}


def split_tcga_by_drug_overlap(
    src: Path,
    out_gdsc: Path,
    out_tcga_only: Path,
) -> dict:
    df = pd.read_csv(src)
    if "drug_name" not in df.columns:
        raise ValueError(f"Expected column drug_name in {src}")

    drug_norm = df["drug_name"].astype(str).str.strip().str.lower()
    mask_gdsc = drug_norm.isin(GDSC_INTERSECT_13)
    mask_tcga = drug_norm.isin(TCGA_ONLY_3)
    unmatched = df.loc[~(mask_gdsc | mask_tcga), "drug_name"].unique()
    if len(unmatched):
        raise ValueError(f"Unclassified drug_name values: {unmatched.tolist()}")

    df_gdsc = df[mask_gdsc].copy()
    df_tcga = df[mask_tcga].copy()
    out_gdsc.parent.mkdir(parents=True, exist_ok=True)
    out_tcga_only.parent.mkdir(parents=True, exist_ok=True)
    df_gdsc.to_csv(out_gdsc, index=False)
    df_tcga.to_csv(out_tcga_only, index=False)

    return {
        "source_rows": len(df),
        "gdsc_intersect_rows": len(df_gdsc),
        "tcga_only_rows": len(df_tcga),
        "gdsc_drugs": sorted(df_gdsc["drug_name"].str.lower().unique()),
        "tcga_only_drugs": sorted(df_tcga["drug_name"].str.lower().unique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split TCGA intersect_pretrain by GDSC drug overlap")
    parser.add_argument(
        "--input",
        default="data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv",
    )
    parser.add_argument(
        "--out-gdsc-intersect",
        default="data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv",
    )
    parser.add_argument(
        "--out-tcga-only",
        default="data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv",
    )
    args = parser.parse_args()
    stats = split_tcga_by_drug_overlap(Path(args.input), Path(args.out_gdsc_intersect), Path(args.out_tcga_only))
    print(f"Source rows: {stats['source_rows']}")
    print(f"GDSC intersect13: {stats['gdsc_intersect_rows']} rows")
    print(f"TCGA only3: {stats['tcga_only_rows']} rows")
    print(f"GDSC drugs: {stats['gdsc_drugs']}")
    print(f"TCGA-only drugs: {stats['tcga_only_drugs']}")
    print(f"Wrote: {args.out_gdsc_intersect}")
    print(f"Wrote: {args.out_tcga_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
