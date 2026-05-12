#!/usr/bin/env python3
"""Build a unified PDX drug-response table from per-drug folders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


PDX_DIR = Path("/workspace/DAPL_git/data/PDX")
MAPPING_PATH = Path("/workspace/DAPL_git/data/0_Drug_old_table/pdtc_gdsc_drug_mapping_all_addsmiles.csv")
OUTPUT_PATH = Path("/workspace/DAPL_git/data/PDX/PDX_drug_response_from_DAPL.csv")
MISSING_CODES_PATH = Path("/workspace/DAPL_git/data/PDX/PDX_drug_response_missing_mapping_codes.csv")


def load_mapping() -> Dict[str, dict]:
    mapping = pd.read_csv(MAPPING_PATH)
    mapping["code"] = mapping["Unnamed: 0"].astype(str).str.strip().str.lower()
    mapping = mapping.drop_duplicates(subset=["code"], keep="first")
    return mapping.set_index("code").to_dict(orient="index")


def build_table() -> tuple[pd.DataFrame, List[str]]:
    code_to_info = load_mapping()
    rows: List[dict] = []
    missing_codes: List[str] = []

    for folder in sorted(PDX_DIR.glob("*data")):
        if not folder.is_dir():
            continue
        code = folder.name[:-4].strip().lower()
        label_path = folder / "pdtclabel.csv"
        if not label_path.exists():
            continue

        label_df = pd.read_csv(label_path)
        if "Unnamed: 0" not in label_df.columns or "AUC" not in label_df.columns:
            raise ValueError(f"Unexpected label format: {label_path}")

        if code not in code_to_info:
            missing_codes.append(code)
            continue

        info = code_to_info[code]
        gdsc_name = info.get("gdsc_name", "")
        mapped_drug_name = info.get("drug_name", "")
        smiles = info.get("smiles", "")

        for _, rec in label_df.iterrows():
            rows.append(
                {
                    "Sample_id": str(rec["Unnamed: 0"]).strip(),
                    "drug_name": gdsc_name,
                    "Label": rec["AUC"],
                    "drug_code": code,
                    "mapped_drug_name": mapped_drug_name,
                    "smiles": smiles,
                }
            )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["Label"] = pd.to_numeric(out_df["Label"], errors="coerce")
    return out_df, sorted(set(missing_codes))


def main() -> None:
    out_df, missing_codes = build_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame({"missing_code": missing_codes}).to_csv(MISSING_CODES_PATH, index=False)

    print(f"output: {OUTPUT_PATH}")
    print(f"shape: {out_df.shape}")
    if not out_df.empty:
        print(f"unique_samples: {out_df['Sample_id'].nunique()}")
        print(f"unique_drugs: {out_df['drug_name'].nunique()}")
    print(f"missing_mapping_codes_count: {len(missing_codes)}")
    print(f"missing_codes_file: {MISSING_CODES_PATH}")


if __name__ == "__main__":
    main()
