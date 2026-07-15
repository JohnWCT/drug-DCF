#!/usr/bin/env python3
"""Cancer-type (primary_disease) mapping for Round 19E cancer-type-held-out splits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CCLE_SAMPLE_INFO = "data/ccle_sample_info_df.csv"


def _load_ccle_sample_info(path: str) -> pd.DataFrame:
    si = pd.read_csv(path)
    if "Unnamed: 0" in si.columns and "ModelID" not in si.columns:
        si = si.rename(columns={"Unnamed: 0": "ModelID"})
    if "ModelID" not in si.columns:
        raise KeyError(f"{path} missing ModelID")
    si["ModelID"] = si["ModelID"].astype(str)
    return si


def _ccle_disease_to_code_map(dev_mid_disease: pd.DataFrame, ccle: pd.DataFrame) -> Dict[str, str]:
    """Majority GDSC primary_disease code per CCLE cancer_type (exclude UNCLASSIFIED)."""
    j = dev_mid_disease.merge(
        ccle[["ModelID", "cancer_type", "primary_disease"]],
        on="ModelID",
        how="inner",
        suffixes=("_dev", "_ccle"),
    )
    j = j[j["primary_disease_dev"].notna()]
    mapping: Dict[str, str] = {}
    for ctype, g in j.groupby(j["cancer_type"].astype(str)):
        codes = g["primary_disease_dev"].astype(str)
        codes = codes[codes != "UNCLASSIFIED"]
        if codes.empty:
            codes = g["primary_disease_dev"].astype(str)
        if codes.empty:
            continue
        mapping[str(ctype)] = str(codes.value_counts().index[0])
    return mapping


def build_modelid_cancer_type_map(
    development: pd.DataFrame,
    *,
    ccle_sample_info_path: str = DEFAULT_CCLE_SAMPLE_INFO,
    disease_column: str = "primary_disease",
) -> Tuple[pd.DataFrame, dict]:
    """
    Each ModelID -> exactly one cancer_type (GDSC-style code).

    Missing primary_disease is fail-fast unless resolvable via CCLE majority map
    (e.g. ACH-000708 Colon/Colorectal Cancer -> COREAD). Never invent UNKNOWN.
    """
    if "ModelID" not in development.columns:
        raise KeyError("ModelID")
    if disease_column not in development.columns:
        raise KeyError(disease_column)

    mid = (
        development[["ModelID", disease_column]]
        .drop_duplicates()
        .rename(columns={disease_column: "primary_disease_raw"})
    )
    mid["ModelID"] = mid["ModelID"].astype(str)

    # Detect multi-disease ModelIDs
    n_per = mid.groupby("ModelID")["primary_disease_raw"].nunique(dropna=False)
    multi = n_per[n_per > 1]
    if len(multi):
        raise AssertionError(
            f"ModelIDs with multiple primary_disease values: {list(multi.index[:10])}"
        )

    ccle = _load_ccle_sample_info(ccle_sample_info_path)
    # Collapse to one row per ModelID from development perspective
    one = mid.drop_duplicates("ModelID").copy()
    code_map = _ccle_disease_to_code_map(
        one.rename(columns={"primary_disease_raw": "primary_disease_dev"})[
            ["ModelID", "primary_disease_dev"]
        ],
        ccle,
    )

    imputed = []
    cancer_types = []
    sources = []
    for _, row in one.iterrows():
        mid_s = str(row["ModelID"])
        raw = row["primary_disease_raw"]
        if pd.notna(raw) and str(raw).strip():
            cancer_types.append(str(raw).strip())
            sources.append("development_primary_disease")
            continue
        ccle_row = ccle[ccle["ModelID"] == mid_s]
        if ccle_row.empty:
            raise AssertionError(f"Missing cancer_type for ModelID={mid_s} (no CCLE row)")
        ctype = str(ccle_row.iloc[0].get("cancer_type") or ccle_row.iloc[0].get("primary_disease") or "")
        if not ctype or ctype == "nan":
            raise AssertionError(f"Missing cancer_type for ModelID={mid_s} (empty CCLE disease)")
        code = code_map.get(ctype)
        if not code:
            raise AssertionError(
                f"Cannot map CCLE disease {ctype!r} to GDSC code for ModelID={mid_s}"
            )
        cancer_types.append(code)
        sources.append(f"ccle_impute:{ctype}->{code}")
        imputed.append({"ModelID": mid_s, "ccle_cancer_type": ctype, "cancer_type": code})

    one["cancer_type"] = cancer_types
    one["mapping_source"] = sources
    if one["cancer_type"].isna().any() or (one["cancer_type"].astype(str).str.len() == 0).any():
        raise AssertionError("cancer_type still missing after imputation")
    if (one["cancer_type"].astype(str) == "UNKNOWN").any():
        raise AssertionError("UNKNOWN cancer_type is forbidden")

    meta = {
        "n_modelids": int(one["ModelID"].nunique()),
        "n_cancer_types": int(one["cancer_type"].nunique()),
        "n_imputed": int(len(imputed)),
        "imputed": imputed,
        "cancer_type_counts": one["cancer_type"].value_counts().to_dict(),
    }
    return one[["ModelID", "cancer_type", "mapping_source", "primary_disease_raw"]], meta


def attach_cancer_type(
    development: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    m = mapping.set_index("ModelID")["cancer_type"].to_dict()
    out = development.copy()
    out["cancer_type"] = out["ModelID"].astype(str).map(m)
    if out["cancer_type"].isna().any():
        miss = out.loc[out["cancer_type"].isna(), "ModelID"].unique()[:10]
        raise AssertionError(f"Unmapped ModelIDs: {list(miss)}")
    return out


def write_cancer_type_group_table(
    root: Path,
    *,
    ccle_sample_info_path: str = DEFAULT_CCLE_SAMPLE_INFO,
) -> Path:
    root = Path(root)
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(splits / "development_rows.csv")
    mapping, meta = build_modelid_cancer_type_map(
        development, ccle_sample_info_path=ccle_sample_info_path
    )
    # Expand to row-level support table
    tagged = attach_cancer_type(development, mapping)
    rows = []
    for ctype, g in tagged.groupby("cancer_type"):
        y = g["Label"].astype(int)
        rows.append(
            {
                "cancer_type": str(ctype),
                "n_rows": int(len(g)),
                "n_modelids": int(g["ModelID"].nunique()),
                "n_drugs": int(g["DRUG_NAME"].nunique()),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "positive_rate": float(y.mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("cancer_type")
    path = splits / "round19e_cancer_type_group_table.csv"
    table.to_csv(path, index=False)
    map_path = splits / "round19e_modelid_cancer_type_map.csv"
    mapping.to_csv(map_path, index=False)
    meta["paths"] = {"group_table": str(path), "modelid_map": str(map_path)}
    (splits / "round19e_cancer_type_group_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument("--ccle-sample-info", default=DEFAULT_CCLE_SAMPLE_INFO)
    args = parser.parse_args()
    path = write_cancer_type_group_table(
        Path(args.root), ccle_sample_info_path=args.ccle_sample_info
    )
    print(json.dumps({"written": str(path)}, indent=2))


if __name__ == "__main__":
    main()
