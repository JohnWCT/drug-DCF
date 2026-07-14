"""Round 19 CV split helpers (ModelID / drug / scaffold / cancer-type)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

from tools.round18_cv_splits import build_grouped_cv_assignments
from tools.round19_scaffold_groups import build_scaffold_map, murcko_scaffold_id


def link_or_reuse_round18_splits(round18_outdir: str, round19_outdir: str) -> Dict[str, str]:
    src = Path(round18_outdir) / "splits"
    dst = Path(round19_outdir) / "splits"
    dst.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for name in [
        "screening_3fold_assignments.csv",
        "formal_5fold_assignments.csv",
        "internal_test_split.csv",
        "development_rows.csv",
        "split_metadata.json",
    ]:
        s = src / name
        d = dst / name
        if not s.is_file():
            raise FileNotFoundError(s)
        if d.exists() or d.is_symlink():
            if d.is_symlink() or d.is_file():
                d.unlink()
            else:
                shutil.rmtree(d)
        try:
            d.symlink_to(s.resolve())
        except OSError:
            shutil.copy2(s, d)
        mapping[name] = str(d)
    return mapping


def link_or_reuse_round18_eligible(round18_outdir: str, round19_outdir: str) -> str:
    src = Path(round18_outdir) / "data" / "round18_eligible_response.csv"
    dst_dir = Path(round19_outdir) / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "round19_eligible_response.csv"
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def build_heldout_assignments(
    eligible_df: pd.DataFrame,
    *,
    group_column: str,
    n_splits: int = 5,
    split_seed: int = 52,
    cv_name: str = "drug_heldout_5fold",
    label_column: str = "Label",
) -> pd.DataFrame:
    """Generic stratified-group held-out CV over an arbitrary group column."""
    return build_grouped_cv_assignments(
        eligible_df,
        n_splits=n_splits,
        group_column=group_column,
        label_column=label_column,
        split_seed=split_seed,
        cv_name=cv_name,
    )


def attach_scaffold_ids(
    eligible_df: pd.DataFrame,
    drug_to_smiles: Dict[str, str],
    *,
    drug_column: str = "DRUG_NAME",
) -> pd.DataFrame:
    df = eligible_df.copy()
    smap = build_scaffold_map(drug_to_smiles)
    df["scaffold_id"] = df[drug_column].astype(str).map(
        lambda d: smap[d] if d in smap else murcko_scaffold_id(drug_to_smiles[d])
    )
    return df
