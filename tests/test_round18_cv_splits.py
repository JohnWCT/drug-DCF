import numpy as np
import pandas as pd
import pytest

from tools.round18_cv_splits import (
    build_grouped_cv_assignments,
    build_internal_test_and_development,
    build_split_qc_reports,
)


def _toy_response(n_models=30, drugs_per=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    rid = 0
    for m in range(n_models):
        mid = f"ACH-{m:06d}"
        for d in range(drugs_per):
            label = int(rng.rand() > 0.7)
            rows.append(
                {
                    "_row_id": rid,
                    "ModelID": mid,
                    "Label": label,
                    "mapped_name": f"drug_{d}",
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def test_internal_test_disjoint_and_reproducible():
    df = _toy_response()
    t1, d1, m1 = build_internal_test_and_development(df, split_seed=42)
    t2, d2, m2 = build_internal_test_and_development(df, split_seed=42)
    assert set(t1["ModelID"]).isdisjoint(set(d1["ModelID"]))
    assert list(t1["_row_id"]) == list(t2["_row_id"])
    assert m1["n_internal_test_rows"] == m2["n_internal_test_rows"]
    assert 0.05 <= m1["internal_test_row_fraction"] <= 0.25


def test_screening_folds_no_modelid_overlap():
    df = _toy_response()
    _, development, _ = build_internal_test_and_development(df, split_seed=42)
    assigns = build_grouped_cv_assignments(development, n_splits=3, split_seed=42)
    for fold_id in assigns["fold_id"].unique():
        fold = assigns[assigns["fold_id"] == fold_id]
        train = set(fold.loc[fold["split_role"] == "train", "ModelID"])
        val = set(fold.loc[fold["split_role"] == "val", "ModelID"])
        assert train.isdisjoint(val)


def test_qc_fails_on_overlap():
    df = _toy_response()
    internal, development, _ = build_internal_test_and_development(df, split_seed=42)
    screening = build_grouped_cv_assignments(development, n_splits=3, split_seed=42, cv_name="screening_3fold")
    formal = build_grouped_cv_assignments(development, n_splits=5, split_seed=42, cv_name="formal_5fold")
    bad_dev = pd.concat([development, internal.head(1)], ignore_index=True)
    with pytest.raises(AssertionError):
        build_split_qc_reports(internal, bad_dev, screening, formal)
