#!/usr/bin/env python3
"""Build TCGA CSVs that keep only patients with available pretrain expression.

Patient IDs are normalized to TCGA patient-level barcode (first 12 chars),
for example:
    TCGA-D3-A1QA-07 -> TCGA-D3-A1QA
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd


def normalize_tcga_patient_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.slice(0, 12)


def get_pretrain_patient_set(pretrain_csv: Path, id_col: str) -> Set[str]:
    pretrain_df = pd.read_csv(pretrain_csv)
    if id_col not in pretrain_df.columns:
        raise ValueError(f"Column '{id_col}' not found in {pretrain_csv}")
    return set(normalize_tcga_patient_id(pretrain_df[id_col]).unique())


def filter_by_pretrain_patients(
    input_csv: Path,
    output_csv: Path,
    patient_col: str,
    pretrain_patients: Set[str],
) -> None:
    df = pd.read_csv(input_csv)
    if patient_col not in df.columns:
        raise ValueError(f"Column '{patient_col}' not found in {input_csv}")

    patient_norm = normalize_tcga_patient_id(df[patient_col])
    keep_mask = patient_norm.isin(pretrain_patients)
    filtered = df.loc[keep_mask].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    input_unique = set(patient_norm.unique())
    overlap = input_unique & pretrain_patients
    missing_exp = input_unique - pretrain_patients
    print(f"[source] {input_csv}")
    print(
        f"  unique_patients={len(input_unique)} overlap_with_pretrain={len(overlap)} "
        f"missing_expression={len(missing_exp)}"
    )
    print(f"  rows_before={len(df)} rows_after={len(filtered)}")
    print(f"[output] {output_csv}")
    print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-csv",
        default="/workspace/DAPL_git/data/TCGA/pretrain_tcga.csv",
        help="Pretrain expression CSV path.",
    )
    parser.add_argument(
        "--pretrain-id-col",
        default="Unnamed: 0",
        help="Patient/sample ID column name in pretrain CSV.",
    )
    parser.add_argument(
        "--ad-csv",
        default="/workspace/DAPL_git/data/TCGA/PMID27354694_DR_OMICS_ad.csv",
        help="Drug response CSV (ad) path.",
    )
    parser.add_argument(
        "--ad-patient-col",
        default="Patient_id",
        help="Patient ID column name in ad CSV.",
    )
    parser.add_argument(
        "--ad-output",
        default="/workspace/DAPL_git/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv",
        help="Output path for ad intersection CSV.",
    )
    parser.add_argument(
        "--dys-csv",
        default="/workspace/DAPL_git/data/TCGA/DiSyn_TCGA_drug_response_from_DAPL.csv",
        help="Drug response CSV (DiSyn) path.",
    )
    parser.add_argument(
        "--dys-patient-col",
        default="Patient_id",
        help="Patient ID column name in DiSyn CSV.",
    )
    parser.add_argument(
        "--dys-output",
        default="/workspace/DAPL_git/data/TCGA/DiSyn_TCGA_drug_response_from_DAPL_intersect_pretrain.csv",
        help="Output path for DiSyn intersection CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretrain_patients = get_pretrain_patient_set(
        pretrain_csv=Path(args.pretrain_csv),
        id_col=args.pretrain_id_col,
    )
    print(
        f"[pretrain] {args.pretrain_csv} | unique_patient_ids={len(pretrain_patients)} "
        f"(normalized to first 12 chars)"
    )
    print("=" * 80)

    filter_by_pretrain_patients(
        input_csv=Path(args.ad_csv),
        output_csv=Path(args.ad_output),
        patient_col=args.ad_patient_col,
        pretrain_patients=pretrain_patients,
    )
    filter_by_pretrain_patients(
        input_csv=Path(args.dys_csv),
        output_csv=Path(args.dys_output),
        patient_col=args.dys_patient_col,
        pretrain_patients=pretrain_patients,
    )


if __name__ == "__main__":
    main()
