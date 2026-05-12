#!/usr/bin/env python3
"""Replace GDSC expression row IDs with CCLE ACH-style Model IDs.

Usage example:
python tools/replace_gdsc_rowid_with_modelid.py \
  --gdsc data/GDSC_EXP_1363.csv \
  --ccle data/ccle_sample_info_df.csv \
  --output data/GDSC_EXP_1363_modelid.csv \
  --unmatched-output data/GDSC_EXP_1363_unmatched_row_ids.txt
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List

import pandas as pd


def normalize_cell_line_name(name: str) -> str:
    """Normalize a cell line name for robust matching."""
    return re.sub(r"[^A-Z0-9]", "", str(name).strip().upper())


def pick_model_id_column(ccle_df: pd.DataFrame) -> str:
    """Choose the ACH-style ID column from CCLE file."""
    candidates = ["ModelID", "model_id", "DepMap_ID", "Unnamed: 0"]
    for col in candidates:
        if col in ccle_df.columns:
            return col
    raise ValueError(
        "No model-id column found in CCLE file. "
        "Expected one of: ModelID, model_id, DepMap_ID, Unnamed: 0"
    )


def build_mapping(ccle_df: pd.DataFrame, model_id_col: str) -> Dict[str, str]:
    """Build stripped_cell_line_name -> model_id map."""
    if "stripped_cell_line_name" not in ccle_df.columns:
        raise ValueError("CCLE file must contain 'stripped_cell_line_name' column.")

    mapping: Dict[str, str] = {}
    duplicates: Dict[str, set] = {}
    src = ccle_df[["stripped_cell_line_name", model_id_col]].dropna()

    for _, row in src.iterrows():
        key = normalize_cell_line_name(row["stripped_cell_line_name"])
        model_id = str(row[model_id_col]).strip()
        if not key:
            continue
        if key in mapping and mapping[key] != model_id:
            duplicates.setdefault(key, {mapping[key]}).add(model_id)
            continue
        mapping[key] = model_id

    if duplicates:
        dup_examples = ", ".join(
            f"{k}:{sorted(v)}" for k, v in list(duplicates.items())[:5]
        )
        raise ValueError(
            "Ambiguous mapping detected for stripped_cell_line_name keys. "
            f"Examples: {dup_examples}"
        )

    return mapping


def convert_row_ids(gdsc_df: pd.DataFrame, mapping: Dict[str, str]) -> tuple[pd.DataFrame, List[str]]:
    """Replace first-column row IDs in GDSC with mapped model IDs."""
    first_col = gdsc_df.columns[0]
    original_row_ids = gdsc_df[first_col].astype(str)
    stripped_names = original_row_ids.str.split("__", n=1).str[0]
    normalized_names = stripped_names.map(normalize_cell_line_name)

    new_ids = normalized_names.map(mapping)
    unmatched = original_row_ids[new_ids.isna()].tolist()

    converted_df = gdsc_df.copy()
    converted_df[first_col] = new_ids.where(new_ids.notna(), original_row_ids)
    return converted_df, unmatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdsc", required=True, help="Path to GDSC_EXP_1363.csv")
    parser.add_argument("--ccle", required=True, help="Path to ccle_sample_info_df.csv")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument(
        "--unmatched-output",
        required=True,
        help="Path to save unmatched original row IDs (txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gdsc_df = pd.read_csv(args.gdsc)
    ccle_df = pd.read_csv(args.ccle)

    model_id_col = pick_model_id_column(ccle_df)
    mapping = build_mapping(ccle_df, model_id_col)
    converted_df, unmatched = convert_row_ids(gdsc_df, mapping)

    converted_df.to_csv(args.output, index=False)
    with open(args.unmatched_output, "w", encoding="utf-8") as f:
        for row_id in unmatched:
            f.write(f"{row_id}\n")

    print(f"Model ID column used: {model_id_col}")
    print(f"Total rows: {len(gdsc_df)}")
    print(f"Matched rows: {len(gdsc_df) - len(unmatched)}")
    print(f"Unmatched rows: {len(unmatched)}")
    print(f"Output CSV: {args.output}")
    print(f"Unmatched list: {args.unmatched_output}")


if __name__ == "__main__":
    main()
