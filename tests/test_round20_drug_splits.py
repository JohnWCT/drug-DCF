from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.round20_drug_splits import audit_drug_identity_table, build_repeated_drug_held_out_splits

ROOT = Path(__file__).resolve().parents[1]
DRUG_TABLE = ROOT / "result/optimization_runs/round19_factorial/splits/round19e_drug_group_table.csv"


def test_drug_identity_audit_ok(tmp_path: Path) -> None:
    out = tmp_path / "drug_identity_audit.json"
    report = audit_drug_identity_table(DRUG_TABLE, out=out)
    assert report["ok"] is True
    assert report["n_drugs"] >= 100
    assert (tmp_path / "drug_identity_mapping.csv").is_file()


def test_repeated_splits_disjoint(tmp_path: Path) -> None:
    table = pd.read_csv(DRUG_TABLE)
    # Synthetic eligible rows: 4 rows per drug for first 20 auc-valid drugs.
    drugs = table.loc[table["auc_valid"].astype(bool), "normalized_drug_id"].head(20).tolist()
    rows = []
    rid = 0
    for d in drugs:
        for i in range(4):
            rows.append({"row_id": rid, "drug_group_id": d, "Label": i % 2})
            rid += 1
    df = pd.DataFrame(rows)
    outdir = tmp_path / "splits"
    result = build_repeated_drug_held_out_splits(
        df,
        drug_group_column="drug_group_id",
        label_column="Label",
        split_seeds=[52, 62, 72],
        n_splits=5,
        outdir=outdir,
    )
    assert result["leakage"]["ok"] is True
    for seed in (52, 62, 72):
        assign = pd.read_csv(outdir / f"drug_held_out_seed{seed}_assignments.csv")
        for fold_id, g in assign.groupby("fold_id"):
            train = set(g.loc[g["split_role"] == "train", "drug_group_id"])
            val = set(g.loc[g["split_role"] == "val", "drug_group_id"])
            assert train.isdisjoint(val)
