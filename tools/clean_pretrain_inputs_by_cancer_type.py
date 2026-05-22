"""
Remove samples whose cancer_type is listed in pretrain_cancer_type_exclude.json
from pretrain input matrices and drug-response tables.

After running this script, pretrain_VAEwC / pretrain_AEwC / pretrain_CVAE can use the
cleaned files directly without runtime cancer_type filtering.

Usage (Docker):
  docker exec -w /workspace/DAPL DAPL python3 tools/clean_pretrain_inputs_by_cancer_type.py \\
    --config config/pretrain_cancer_type_exclude.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Set, Tuple

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.pretrain_common import (
    DEFAULT_CANCER_TYPE_EXCLUDE_CONFIG,
    is_trainable_cancer_type,
    load_cancer_type_exclude_set,
    tcga_three_segment_key,
)

DEFAULT_CCLE_INFO = os.path.join("data", "ccle_sample_info_df.csv")
DEFAULT_TCGA_INFO = os.path.join("data", "TCGA", "xena_sample_info_df.csv")
DEFAULT_PDX_REFERENCE = os.path.join("data", "PDX", "PDX_target_cancer_reference.csv")

DEFAULT_TARGETS = {
    "pretrain_ccle": os.path.join("data", "pretrain_ccle.csv"),
    "pretrain_tcga": os.path.join("data", "TCGA", "pretrain_tcga.csv"),
    "tcga_drug_response": os.path.join("data", "TCGA", "TCGA_drug_response_from_DAPL.csv"),
    "tcga_omics_intersect": os.path.join("data", "TCGA", "PMID27354694_DR_OMICS_ad_intersect_pretrain.csv"),
    "gdsc2_dose_response": os.path.join("data", "GDSC2_fitted_dose_response_MaxScreen_raw.csv"),
}


def build_excluded_id_sets(
    exclude_config: str,
    ccle_info_path: str = DEFAULT_CCLE_INFO,
    tcga_info_path: str = DEFAULT_TCGA_INFO,
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Return (excluded_ccle_ids, excluded_tcga_sample_ids, excluded_tcga_patient_ids, exclude_tokens)."""
    exclude_set = load_cancer_type_exclude_set(exclude_config)

    ccle_info = pd.read_csv(ccle_info_path, index_col=0)
    ccle_info.index = ccle_info.index.astype(str)
    if "cancer_type" not in ccle_info.columns:
        raise ValueError(f"Missing cancer_type in {ccle_info_path}")
    ccle_mask = ~ccle_info["cancer_type"].map(lambda v: is_trainable_cancer_type(v, exclude_set))
    excluded_ccle = set(ccle_info.index[ccle_mask].astype(str))

    xena_info = pd.read_csv(tcga_info_path, index_col=0)
    xena_info.index = xena_info.index.astype(str)
    if "cancer_type" not in xena_info.columns:
        raise ValueError(f"Missing cancer_type in {tcga_info_path}")
    xena_mask = ~xena_info["cancer_type"].map(lambda v: is_trainable_cancer_type(v, exclude_set))
    excluded_tcga_samples = set(xena_info.index[xena_mask].astype(str))
    xena_info["_patient_id"] = xena_info.index.map(tcga_three_segment_key)
    excluded_tcga_patients = set(xena_info.loc[xena_mask, "_patient_id"].astype(str).unique())

    return excluded_ccle, excluded_tcga_samples, excluded_tcga_patients, exclude_set


def _filter_indexed_matrix(path: str, exclude_ids: Set[str], inplace: bool) -> Dict:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    before = len(df)
    keep_mask = ~df.index.isin(exclude_ids)
    df = df.loc[keep_mask]
    after = len(df)
    if inplace:
        df.to_csv(path)
    return {"file": path, "before": before, "after": after, "removed": before - after}


def _filter_tcga_patients_table(
    path: str,
    patient_cols: Tuple[str, ...],
    exclude_patients: Set[str],
    inplace: bool,
) -> Dict:
    df = pd.read_csv(path)
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in patient_cols:
        if col in df.columns:
            mask &= ~df[col].astype(str).isin(exclude_patients)
    df = df.loc[mask]
    after = len(df)
    if inplace:
        df.to_csv(path, index=False)
    return {"file": path, "before": before, "after": after, "removed": before - after}


def _filter_gdsc_by_model(path: str, exclude_model_ids: Set[str], inplace: bool) -> Dict:
    df = pd.read_csv(path)
    before = len(df)
    if "ModelID" not in df.columns:
        raise ValueError(f"Missing ModelID column in {path}")
    keep = ~df["ModelID"].astype(str).isin(exclude_model_ids)
    df = df.loc[keep]
    after = len(df)
    if inplace:
        df.to_csv(path, index=False)
    return {"file": path, "before": before, "after": after, "removed": before - after}


def normalize_pdx_reference(path: str = DEFAULT_PDX_REFERENCE, inplace: bool = True) -> Dict:
    """Rename cancerType -> cancer_type for consistency with CCLE/TCGA sample info."""
    df = pd.read_csv(path)
    changed = False
    if "cancerType" in df.columns and "cancer_type" not in df.columns:
        df = df.rename(columns={"cancerType": "cancer_type"})
        changed = True
    elif "cancerType" in df.columns and "cancer_type" in df.columns:
        df["cancer_type"] = df["cancer_type"].fillna(df["cancerType"])
        df = df.drop(columns=["cancerType"])
        changed = True
    if inplace and changed:
        df.to_csv(path, index=False)
    return {"file": path, "renamed_cancer_type_column": changed, "columns": list(df.columns)}


def run_cleanup(
    exclude_config: str = DEFAULT_CANCER_TYPE_EXCLUDE_CONFIG,
    targets: Dict[str, str] = None,
    ccle_info_path: str = DEFAULT_CCLE_INFO,
    tcga_info_path: str = DEFAULT_TCGA_INFO,
    pdx_reference_path: str = DEFAULT_PDX_REFERENCE,
    inplace: bool = True,
    normalize_pdx: bool = True,
) -> Dict:
    targets = targets or DEFAULT_TARGETS
    excluded_ccle, excluded_tcga_samples, excluded_tcga_patients, exclude_set = build_excluded_id_sets(
        exclude_config, ccle_info_path, tcga_info_path
    )
    print(f"[exclude] tokens: {sorted(exclude_set)}")
    print(f"[exclude] CCLE cell lines: {len(excluded_ccle)}")
    print(f"[exclude] TCGA samples: {len(excluded_tcga_samples)}")
    print(f"[exclude] TCGA patients: {len(excluded_tcga_patients)}")

    logs = {
        "exclude_config": exclude_config,
        "exclude_tokens": sorted(exclude_set),
        "excluded_ccle_count": len(excluded_ccle),
        "excluded_tcga_sample_count": len(excluded_tcga_samples),
        "excluded_tcga_patient_count": len(excluded_tcga_patients),
        "files": [],
    }

    if normalize_pdx and os.path.exists(pdx_reference_path):
        logs["files"].append(normalize_pdx_reference(pdx_reference_path, inplace=inplace))

    if os.path.exists(targets["pretrain_ccle"]):
        logs["files"].append(_filter_indexed_matrix(targets["pretrain_ccle"], excluded_ccle, inplace))

    if os.path.exists(targets["pretrain_tcga"]):
        logs["files"].append(_filter_indexed_matrix(targets["pretrain_tcga"], excluded_tcga_samples, inplace))

    if os.path.exists(targets["tcga_drug_response"]):
        logs["files"].append(
            _filter_tcga_patients_table(
                targets["tcga_drug_response"],
                ("Patient_id", "patient"),
                excluded_tcga_patients,
                inplace,
            )
        )

    if os.path.exists(targets["tcga_omics_intersect"]):
        logs["files"].append(
            _filter_tcga_patients_table(
                targets["tcga_omics_intersect"],
                ("Patient_id", "patient"),
                excluded_tcga_patients,
                inplace,
            )
        )

    if os.path.exists(targets["gdsc2_dose_response"]):
        logs["files"].append(_filter_gdsc_by_model(targets["gdsc2_dose_response"], excluded_ccle, inplace))

    summary_path = os.path.join("result", "clean_pretrain_inputs_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"[done] summary -> {summary_path}")
    return logs


def main():
    parser = argparse.ArgumentParser(description="Clean pretrain inputs by cancer_type exclude config")
    parser.add_argument("--config", default=DEFAULT_CANCER_TYPE_EXCLUDE_CONFIG)
    parser.add_argument("--ccle-info", default=DEFAULT_CCLE_INFO)
    parser.add_argument("--tcga-info", default=DEFAULT_TCGA_INFO)
    parser.add_argument("--pdx-reference", default=DEFAULT_PDX_REFERENCE)
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write files")
    parser.add_argument("--skip-pdx-rename", action="store_true")
    args = parser.parse_args()
    run_cleanup(
        exclude_config=args.config,
        ccle_info_path=args.ccle_info,
        tcga_info_path=args.tcga_info,
        pdx_reference_path=args.pdx_reference,
        inplace=not args.dry_run,
        normalize_pdx=not args.skip_pdx_rename,
    )


if __name__ == "__main__":
    main()
