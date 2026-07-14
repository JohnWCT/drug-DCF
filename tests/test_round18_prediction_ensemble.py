import pandas as pd
import pytest

from tools.round18_prediction_ensemble import (
    assert_fold_prediction_uniqueness,
    ensemble_fold_predictions,
)


def _mk(n_folds=5):
    rows = []
    for fold in range(n_folds):
        rows.append(
            {
                "eval_row_id": "internal_test|M1|DrugA|1|0",
                "Patient_id": "M1",
                "drug_name": "DrugA",
                "Label": 1,
                "fold_id": fold,
                "probability": 0.2 + 0.1 * fold,
                "target_key": "internal_test",
                "logit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_ensemble_mean_and_assertions():
    df = _mk()
    out = ensemble_fold_predictions(df, required_folds=5)
    assert len(out) == 1
    assert abs(float(out.iloc[0]["probability"]) - 0.4) < 1e-9


def test_duplicate_fold_rejected():
    df = _mk()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="unique"):
        assert_fold_prediction_uniqueness(df)


def test_incomplete_folds_rejected():
    df = _mk(n_folds=4)
    with pytest.raises(AssertionError, match="Expected 5 folds"):
        assert_fold_prediction_uniqueness(df)
