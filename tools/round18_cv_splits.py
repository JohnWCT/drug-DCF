"""Round 18 ModelID-grouped CV splits and QC."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _ensure_binary_labels(labels: pd.Series) -> pd.Series:
    vals = set(pd.Series(labels).dropna().astype(int).unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Labels must be binary {{0,1}}, got {sorted(vals)}")
    return labels.astype(int)


def build_internal_test_and_development(
    response_df: pd.DataFrame,
    *,
    group_column: str = "ModelID",
    label_column: str = "Label",
    split_seed: int = 42,
    n_splits: int = 10,
    test_fold: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Use StratifiedGroupKFold fold0 as locked internal test; rest = development."""
    df = response_df.reset_index(drop=True).copy()
    df["_row_id"] = np.arange(len(df), dtype=int)
    y = _ensure_binary_labels(df[label_column])
    groups = df[group_column].astype(str)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    folds = list(splitter.split(df, y, groups))
    if test_fold < 0 or test_fold >= len(folds):
        raise ValueError(f"test_fold={test_fold} out of range for n_splits={n_splits}")

    _, test_idx = folds[test_fold]
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_idx] = True

    internal_test = df.loc[test_mask].copy()
    development = df.loc[~test_mask].copy()

    meta = {
        "internal_test_split_method": f"StratifiedGroupKFold_{n_splits}fold_fold{test_fold}",
        "internal_test_split_seed": split_seed,
        "group_column": group_column,
        "stratification_column": label_column,
        "n_total_rows": int(len(df)),
        "n_internal_test_rows": int(len(internal_test)),
        "n_development_rows": int(len(development)),
        "internal_test_row_fraction": float(len(internal_test) / max(len(df), 1)),
        "n_internal_test_model_ids": int(internal_test[group_column].nunique()),
        "n_development_model_ids": int(development[group_column].nunique()),
    }
    return internal_test, development, meta


def build_grouped_cv_assignments(
    development_df: pd.DataFrame,
    *,
    n_splits: int,
    group_column: str = "ModelID",
    label_column: str = "Label",
    split_seed: int = 42,
    cv_name: str = "screening_3fold",
) -> pd.DataFrame:
    """Assign each development row to a validation fold (train = complementary)."""
    df = development_df.reset_index(drop=True).copy()
    if "_row_id" not in df.columns:
        df["_row_id"] = np.arange(len(df), dtype=int)
    y = _ensure_binary_labels(df[label_column])
    groups = df[group_column].astype(str)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(df, y, groups)):
        train_groups = set(groups.iloc[train_idx].tolist())
        val_groups = set(groups.iloc[val_idx].tolist())
        overlap = train_groups & val_groups
        if overlap:
            raise AssertionError(f"{cv_name} fold {fold_id} ModelID overlap: {sorted(list(overlap))[:5]}")

        for idx in val_idx:
            rows.append(
                {
                    "cv_name": cv_name,
                    "fold_id": int(fold_id),
                    "split_role": "val",
                    "_row_id": int(df.iloc[idx]["_row_id"]),
                    group_column: str(df.iloc[idx][group_column]),
                    label_column: int(df.iloc[idx][label_column]),
                }
            )
        for idx in train_idx:
            rows.append(
                {
                    "cv_name": cv_name,
                    "fold_id": int(fold_id),
                    "split_role": "train",
                    "_row_id": int(df.iloc[idx]["_row_id"]),
                    group_column: str(df.iloc[idx][group_column]),
                    label_column: int(df.iloc[idx][label_column]),
                }
            )
    return pd.DataFrame(rows)


def _subset_summary(
    df: pd.DataFrame,
    *,
    split_name: str,
    fold_id: Optional[int],
    group_column: str,
    label_column: str,
    drug_column: Optional[str],
) -> Dict[str, Any]:
    labels = _ensure_binary_labels(df[label_column])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    out = {
        "split_name": split_name,
        "fold_id": fold_id if fold_id is not None else -1,
        "n_rows": int(len(df)),
        "n_model_ids": int(df[group_column].nunique()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": float(n_pos / max(len(df), 1)),
    }
    if drug_column and drug_column in df.columns:
        out["n_drugs"] = int(df[drug_column].nunique())
    else:
        out["n_drugs"] = -1
    return out


def build_split_qc_reports(
    internal_test: pd.DataFrame,
    development: pd.DataFrame,
    screening_assignments: pd.DataFrame,
    formal_assignments: pd.DataFrame,
    *,
    group_column: str = "ModelID",
    label_column: str = "Label",
    drug_column: str = "mapped_name",
) -> Dict[str, pd.DataFrame]:
    """Produce balance / label / overlap QC tables; fail on any ModelID overlap."""
    balance_rows = [
        _subset_summary(
            internal_test,
            split_name="internal_test",
            fold_id=None,
            group_column=group_column,
            label_column=label_column,
            drug_column=drug_column,
        ),
        _subset_summary(
            development,
            split_name="development",
            fold_id=None,
            group_column=group_column,
            label_column=label_column,
            drug_column=drug_column,
        ),
    ]

    overlap_rows = []
    test_ids = set(internal_test[group_column].astype(str))
    dev_ids = set(development[group_column].astype(str))
    overlap_test_dev = test_ids & dev_ids
    overlap_rows.append(
        {
            "check": "internal_test_vs_development",
            "fold_id": -1,
            "ModelID_overlap_count": int(len(overlap_test_dev)),
            "overlap_example": ",".join(sorted(list(overlap_test_dev))[:5]),
            "status": "fail" if overlap_test_dev else "pass",
        }
    )
    if overlap_test_dev:
        raise AssertionError(f"internal test overlaps development: {sorted(list(overlap_test_dev))[:10]}")

    label_rows = []
    for name, frame in (("internal_test", internal_test), ("development", development)):
        labels = _ensure_binary_labels(frame[label_column])
        label_rows.append(
            {
                "split_name": name,
                "fold_id": -1,
                "n_label_0": int((labels == 0).sum()),
                "n_label_1": int((labels == 1).sum()),
            }
        )

    drug_rows = []
    for name, frame in (("internal_test", internal_test), ("development", development)):
        if drug_column in frame.columns:
            vc = frame[drug_column].astype(str).value_counts()
            for drug, cnt in vc.items():
                drug_rows.append({"split_name": name, "fold_id": -1, "drug": drug, "n_rows": int(cnt)})

    for cv_name, assigns in (
        ("screening_3fold", screening_assignments),
        ("formal_5fold", formal_assignments),
    ):
        for fold_id in sorted(assigns["fold_id"].unique()):
            fold_df = assigns[assigns["fold_id"] == fold_id]
            train_ids = set(fold_df.loc[fold_df["split_role"] == "train", group_column].astype(str))
            val_ids = set(fold_df.loc[fold_df["split_role"] == "val", group_column].astype(str))
            overlap = train_ids & val_ids
            overlap_rows.append(
                {
                    "check": f"{cv_name}_train_vs_val",
                    "fold_id": int(fold_id),
                    "ModelID_overlap_count": int(len(overlap)),
                    "overlap_example": ",".join(sorted(list(overlap))[:5]),
                    "status": "fail" if overlap else "pass",
                }
            )
            if overlap:
                raise AssertionError(f"{cv_name} fold {fold_id} overlap: {sorted(list(overlap))[:10]}")

            # Map val rows back to development for summaries
            val_row_ids = set(fold_df.loc[fold_df["split_role"] == "val", "_row_id"].astype(int))
            train_row_ids = set(fold_df.loc[fold_df["split_role"] == "train", "_row_id"].astype(int))
            val_dev = development[development["_row_id"].isin(val_row_ids)]
            train_dev = development[development["_row_id"].isin(train_row_ids)]
            balance_rows.append(
                _subset_summary(
                    train_dev,
                    split_name=f"{cv_name}_train",
                    fold_id=int(fold_id),
                    group_column=group_column,
                    label_column=label_column,
                    drug_column=drug_column,
                )
            )
            balance_rows.append(
                _subset_summary(
                    val_dev,
                    split_name=f"{cv_name}_val",
                    fold_id=int(fold_id),
                    group_column=group_column,
                    label_column=label_column,
                    drug_column=drug_column,
                )
            )
            for role, sub in (("train", train_dev), ("val", val_dev)):
                labels = _ensure_binary_labels(sub[label_column])
                label_rows.append(
                    {
                        "split_name": f"{cv_name}_{role}",
                        "fold_id": int(fold_id),
                        "n_label_0": int((labels == 0).sum()),
                        "n_label_1": int((labels == 1).sum()),
                    }
                )

    return {
        "fold_balance_report": pd.DataFrame(balance_rows),
        "fold_label_distribution": pd.DataFrame(label_rows),
        "fold_drug_distribution": pd.DataFrame(drug_rows),
        "fold_group_overlap_qc": pd.DataFrame(overlap_rows),
    }


def write_round18_splits(
    response_path: str,
    outdir: str,
    *,
    group_column: str = "ModelID",
    label_column: str = "Label",
    drug_column: str = "mapped_name",
    split_seed: int = 42,
    screening_folds: int = 3,
    formal_folds: int = 5,
) -> Dict[str, str]:
    """Build all Round 18 split artifacts under outdir/splits."""
    out = Path(outdir)
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    response = pd.read_csv(response_path)
    if drug_column not in response.columns:
        for alt in ("mapped_name", "drug_name", "DRUG_NAME"):
            if alt in response.columns:
                drug_column = alt
                break

    internal_test, development, meta = build_internal_test_and_development(
        response,
        group_column=group_column,
        label_column=label_column,
        split_seed=split_seed,
    )
    meta.update(
        {
            "screening_cv_folds": screening_folds,
            "formal_cv_folds": formal_folds,
            "cv_split_seed": split_seed,
            "drug_column": drug_column,
            "response_data_path": str(response_path),
        }
    )

    screening = build_grouped_cv_assignments(
        development,
        n_splits=screening_folds,
        group_column=group_column,
        label_column=label_column,
        split_seed=split_seed,
        cv_name="screening_3fold",
    )
    formal = build_grouped_cv_assignments(
        development,
        n_splits=formal_folds,
        group_column=group_column,
        label_column=label_column,
        split_seed=split_seed,
        cv_name="formal_5fold",
    )

    qc = build_split_qc_reports(
        internal_test,
        development,
        screening,
        formal,
        group_column=group_column,
        label_column=label_column,
        drug_column=drug_column,
    )

    paths = {
        "internal_test_split": str(splits_dir / "internal_test_split.csv"),
        "development_rows": str(splits_dir / "development_rows.csv"),
        "screening_3fold_assignments": str(splits_dir / "screening_3fold_assignments.csv"),
        "formal_5fold_assignments": str(splits_dir / "formal_5fold_assignments.csv"),
        "split_summary": str(splits_dir / "split_summary.csv"),
        "split_metadata": str(splits_dir / "split_metadata.json"),
        "fold_balance_report": str(splits_dir / "fold_balance_report.csv"),
        "fold_drug_distribution": str(splits_dir / "fold_drug_distribution.csv"),
        "fold_label_distribution": str(splits_dir / "fold_label_distribution.csv"),
        "fold_group_overlap_qc": str(splits_dir / "fold_group_overlap_qc.csv"),
    }

    internal_test.to_csv(paths["internal_test_split"], index=False)
    development.to_csv(paths["development_rows"], index=False)
    screening.to_csv(paths["screening_3fold_assignments"], index=False)
    formal.to_csv(paths["formal_5fold_assignments"], index=False)
    qc["fold_balance_report"].to_csv(paths["fold_balance_report"], index=False)
    qc["fold_drug_distribution"].to_csv(paths["fold_drug_distribution"], index=False)
    qc["fold_label_distribution"].to_csv(paths["fold_label_distribution"], index=False)
    qc["fold_group_overlap_qc"].to_csv(paths["fold_group_overlap_qc"], index=False)

    summary = qc["fold_balance_report"].copy()
    summary.to_csv(paths["split_summary"], index=False)
    Path(paths["split_metadata"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Round 18 CV splits")
    parser.add_argument("--response", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    paths = write_round18_splits(args.response, args.outdir, split_seed=args.split_seed)
    print(json.dumps({"ok": True, "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
