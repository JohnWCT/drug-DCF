#!/usr/bin/env python3
"""Prepare or validate unseen-drug split manifest for xa_validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.utils.hashing import sha256_file, sha256_json

DEV_ROWS = ROOT / "result/optimization_runs/round19_factorial/splits/development_rows.csv"
DRUG_TABLE = ROOT / "result/optimization_runs/round19_factorial/splits/round19e_drug_group_table.csv"


def _attach_drug_group(dev: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    name_to_canon = dict(zip(table["DRUG_NAME"].astype(str), table["canonical_smiles"].astype(str)))
    out = dev.copy()
    out["drug_group_id"] = out["DRUG_NAME"].astype(str).map(name_to_canon)
    if out["drug_group_id"].isna().any():
        missing = sorted(out.loc[out["drug_group_id"].isna(), "DRUG_NAME"].unique())[:10]
        raise ValueError(f"Unmapped drugs for drug_group_id: {missing}")
    return out


def _drug_triple_split(groups: List[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.RandomState(int(seed))
    order = list(groups)
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    test_drugs = order[:n_test]
    val_drugs = order[n_test : n_test + n_val]
    train_drugs = order[n_test + n_val :]
    if not train_drugs:
        train_drugs = val_drugs[:-1]
        val_drugs = val_drugs[-1:]
    return train_drugs, val_drugs, test_drugs


def build_real_split_manifest(
    *,
    seeds: list[int],
    dev_rows_path: Path = DEV_ROWS,
    drug_table_path: Path = DRUG_TABLE,
    assignments_out: Path | None = None,
) -> Dict[str, Any]:
    dev = pd.read_csv(dev_rows_path)
    table = pd.read_csv(drug_table_path)
    dev = _attach_drug_group(dev, table)
    dataset_hash = sha256_file(dev_rows_path)

    assignment_rows: List[pd.DataFrame] = []
    splits = []
    groups_all = sorted(dev["drug_group_id"].astype(str).unique())

    for seed in seeds:
        train_drugs, val_drugs, test_drugs = _drug_triple_split(groups_all, seed)
        train_set, val_set, test_set = set(train_drugs), set(val_drugs), set(test_drugs)
        assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)

        def _assign(role: str, drug_set: set[str]) -> pd.DataFrame:
            part = dev[dev["drug_group_id"].astype(str).isin(drug_set)].copy()
            part["split_seed"] = int(seed)
            part["split_role"] = role
            return part[["_row_id", "ModelID", "DRUG_NAME", "drug_group_id", "Label", "split_seed", "split_role"]]

        seed_df = pd.concat(
            [_assign("train", train_set), _assign("val", val_set), _assign("test", test_set)],
            ignore_index=True,
        )
        assignment_rows.append(seed_df)
        splits.append(
            {
                "split_seed": int(seed),
                "train_drug_ids": sorted(train_set),
                "validation_drug_ids": sorted(val_set),
                "test_drug_ids": sorted(test_set),
                "train_sample_count": int((seed_df["split_role"] == "train").sum()),
                "validation_sample_count": int((seed_df["split_role"] == "val").sum()),
                "test_sample_count": int((seed_df["split_role"] == "test").sum()),
                "dataset_hash": dataset_hash,
            }
        )

    assignments = pd.concat(assignment_rows, ignore_index=True)
    if assignments_out is not None:
        assignments_out.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(assignments_out, index=False)

    manifest = {
        "schema": "unseen_drug_split_manifest",
        "version": 1,
        "synthetic": False,
        "splits": splits,
        "manifest_hash": sha256_json(splits),
        "assignments_csv": str(assignments_out) if assignments_out else None,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/biocda/xa_validation.yaml",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_path = ROOT / config["data"]["split_manifest"]
    assignments_path = out_path.parent / "unseen_drug_assignments.csv"
    seeds = config["experiment"]["seeds"]

    if out_path.is_file() and not args.force and not config["data"].get("synthetic_smoke", False):
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
        if not manifest.get("synthetic", True):
            print(f"SPLIT_MANIFEST=exists path={out_path}")
            return

    manifest = build_real_split_manifest(
        seeds=seeds,
        assignments_out=assignments_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path = ROOT / "reports/split_manifest_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "path": str(out_path),
                "assignments_csv": str(assignments_path),
                "n_splits": len(manifest["splits"]),
                "synthetic": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SPLIT_MANIFEST=created path={out_path}")


if __name__ == "__main__":
    main()
