#!/usr/bin/env python3
"""Round 20 repeated drug-held-out splits and drug-identity audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DrugIdentityAuditError(RuntimeError):
    """Raised when drug identity mapping fails leakage or canonicalize checks."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_drug_identity_table(
    drug_table_path: Path | str,
    *,
    out: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(drug_table_path)
    if not path.is_file():
        raise DrugIdentityAuditError(f"Missing drug group table: {path}")
    table = pd.read_csv(path)
    required = {
        "DRUG_NAME",
        "normalized_drug_id",
        "canonical_smiles",
        "n_rows",
        "n_positive",
        "n_negative",
        "auc_valid",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise DrugIdentityAuditError(f"Drug table missing columns: {missing}")

    hard_issues: list[str] = []
    merge_exceptions: list[dict] = []

    # normalized id -> unique canonical smiles (hard fail: one name, multiple molecules)
    for nid, g in table.groupby("normalized_drug_id"):
        smiles = set(g["canonical_smiles"].astype(str))
        if len(smiles) > 1:
            hard_issues.append(f"normalized_id_multi_smiles:{nid}")
        names = set(g["DRUG_NAME"].astype(str))
        if len(names) > 1:
            hard_issues.append(f"normalized_id_multi_alias:{nid}:{sorted(names)}")

    # canonical smiles shared by multiple normalized ids => merge into one drug_group_id
    for smi, g in table.groupby("canonical_smiles"):
        norms = sorted(set(g["normalized_drug_id"].astype(str)))
        names = sorted(set(g["DRUG_NAME"].astype(str)))
        if len(norms) > 1:
            merge_exceptions.append(
                {
                    "canonical_smiles": str(smi),
                    "normalized_drug_ids": norms,
                    "raw_drug_names": names,
                    "action": "merge_to_canonical_smiles_group",
                }
            )

    empty_canon = table["canonical_smiles"].isna() | (
        table["canonical_smiles"].astype(str).str.len() == 0
    )
    n_empty = int(empty_canon.sum())
    if n_empty:
        hard_issues.append(f"empty_canonical_smiles:{n_empty}")

    mapping = table.copy()
    mapping["raw_drug_name"] = mapping["DRUG_NAME"].astype(str)
    mapping["normalized_drug_name"] = mapping["normalized_drug_id"].astype(str)
    mapping["raw_smiles"] = mapping["canonical_smiles"].astype(str)
    # Round 20 group identity is canonical SMILES so aliases cannot leak across splits.
    mapping["drug_group_id"] = mapping["canonical_smiles"].astype(str)
    mapping["scaffold_id"] = None
    mapping["inchikey_if_available"] = None

    report = {
        "schema": "round20_drug_identity_audit",
        "schema_version": 1,
        "source_table": str(path),
        "source_sha256": _sha256_file(path),
        "n_drugs": int(len(table)),
        "n_drug_groups_after_merge": int(mapping["drug_group_id"].nunique()),
        "n_auc_valid": int(table["auc_valid"].astype(bool).sum()),
        "n_empty_canonical_smiles": n_empty,
        "n_alias_merges": len(merge_exceptions),
        "alias_merge_exceptions": merge_exceptions,
        "hard_issues": hard_issues,
        "ok": len(hard_issues) == 0,
        "group_column": "drug_group_id",
        "notes": [
            "drug_group_id = canonical_smiles so alias/salt-form collisions cannot cross train/val.",
            "alias merges are recorded in alias_merge_exceptions; not a hard failure.",
            "scaffold_id / inchikey reserved for future enrichment.",
        ],
    }
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path = out_path.parent / "drug_identity_mapping.csv"
        mapping[
            [
                "raw_drug_name",
                "normalized_drug_name",
                "raw_smiles",
                "canonical_smiles",
                "inchikey_if_available",
                "drug_group_id",
                "scaffold_id",
                "n_rows",
                "n_positive",
                "n_negative",
                "auc_valid",
            ]
        ].to_csv(mapping_path, index=False)
        report["mapping_csv"] = str(mapping_path)
        exc_path = out_path.parent / "drug_identity_exceptions.json"
        exc_path.write_text(
            json.dumps({"alias_merge_exceptions": merge_exceptions}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        report["exceptions_json"] = str(exc_path)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if hard_issues:
        raise DrugIdentityAuditError(
            "Drug identity audit failed: " + "; ".join(hard_issues[:10])
        )
    return report


def build_repeated_drug_held_out_splits(
    eligible_df: pd.DataFrame,
    *,
    drug_group_column: str,
    label_column: str,
    split_seeds: list[int],
    n_splits: int,
    outdir: str | Path,
) -> dict[str, Any]:
    """Build repeated GroupKFold drug-held-out assignments.

    This is Stage 20A infrastructure; Stage 20-0 only needs the function surface
    and identity audit. Full split generation is invoked by later stages.
    """
    from sklearn.model_selection import GroupKFold

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if drug_group_column not in eligible_df.columns:
        raise KeyError(drug_group_column)
    if label_column not in eligible_df.columns:
        raise KeyError(label_column)

    df = eligible_df.copy().reset_index(drop=True)
    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(int)

    all_rows: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for seed in split_seeds:
        # Deterministic group order under seed via shuffled unique groups.
        groups = df[drug_group_column].astype(str)
        unique_groups = sorted(groups.unique())
        rng = pd.Series(unique_groups).sample(frac=1.0, random_state=int(seed)).tolist()
        group_rank = {g: i for i, g in enumerate(rng)}
        order = groups.map(group_rank).to_numpy()
        ordered = df.iloc[order.argsort(kind="mergesort")].reset_index(drop=True)
        gkf = GroupKFold(n_splits=int(n_splits))
        y = ordered[label_column].to_numpy()
        grp = ordered[drug_group_column].astype(str).to_numpy()
        seed_rows = []
        for fold_id, (train_idx, val_idx) in enumerate(gkf.split(ordered, y, grp)):
            train_groups = set(grp[train_idx])
            val_groups = set(grp[val_idx])
            overlap = train_groups & val_groups
            if overlap:
                raise AssertionError(f"seed={seed} fold={fold_id} group overlap: {sorted(overlap)[:5]}")
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise AssertionError(f"seed={seed} fold={fold_id} empty split")
            if len(train_groups) == 0 or len(val_groups) == 0:
                raise AssertionError(f"seed={seed} fold={fold_id} empty drug groups")
            for idx, role in ((train_idx, "train"), (val_idx, "val")):
                part = ordered.iloc[idx][["row_id", drug_group_column]].copy()
                part = part.rename(columns={drug_group_column: "drug_group_id"})
                part["split_seed"] = int(seed)
                part["fold_id"] = int(fold_id)
                part["split_role"] = role
                seed_rows.append(part)
            fold_reports.append(
                {
                    "split_seed": int(seed),
                    "fold_id": int(fold_id),
                    "n_train_rows": int(len(train_idx)),
                    "n_val_rows": int(len(val_idx)),
                    "n_train_drugs": int(len(train_groups)),
                    "n_val_drugs": int(len(val_groups)),
                }
            )
        seed_df = pd.concat(seed_rows, ignore_index=True)
        seed_path = out / f"drug_held_out_seed{seed}_assignments.csv"
        seed_df.to_csv(seed_path, index=False)
        all_rows.append(seed_df)

    leakage = {
        "schema": "round20_drug_split_leakage_audit",
        "ok": True,
        "folds": fold_reports,
        "rules": [
            "train drug_group_id ∩ val drug_group_id = ∅",
            "assignments generated by GroupKFold on drug_group_id",
        ],
    }
    (out / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    meta = {
        "schema": "round20_drug_held_out_metadata",
        "split_seeds": [int(s) for s in split_seeds],
        "n_splits": int(n_splits),
        "group_column": drug_group_column,
        "label_column": label_column,
        "n_rows": int(len(df)),
        "n_groups": int(df[drug_group_column].nunique()),
    }
    (out / "drug_held_out_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": meta, "leakage": leakage, "outdir": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit-identity", help="Audit Round 19E drug identity table")
    audit_p.add_argument(
        "--drug-table",
        default=str(
            PROJECT_ROOT
            / "result/optimization_runs/round19_factorial/splits/round19e_drug_group_table.csv"
        ),
    )
    audit_p.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT
            / "result/optimization_runs/round20_unseen_drug_closure/audit/drug_identity_audit.json"
        ),
    )
    args = parser.parse_args()
    if args.command == "audit-identity":
        report = audit_drug_identity_table(args.drug_table, out=args.out)
        print(json.dumps({"ok": report["ok"], "n_drugs": report["n_drugs"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
