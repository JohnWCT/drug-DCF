import pandas as pd
import pytest

from tools.round18_prediction_ensemble import (
    assert_no_best_fold_selection,
    ensemble_fold_predictions,
)


def _fold_df(n_folds=5):
    rows = []
    for fold in range(n_folds):
        rows.append(
            {
                "target_key": "gdsc_intersect13",
                "Patient_id": "P1",
                "drug_name": "cisplatin",
                "Label": 1,
                "fold_id": fold,
                "logit": 0.1 * fold,
                "probability": 0.2 + 0.05 * fold,
            }
        )
    return pd.DataFrame(rows)


def test_ensemble_requires_all_five_folds():
    df = _fold_df(5)
    out = ensemble_fold_predictions(df, required_folds=5)
    assert len(out) == 1
    assert out.iloc[0]["n_folds"] == 5
    assert abs(out.iloc[0]["probability"] - df["probability"].mean()) < 1e-8


def test_ensemble_fails_when_fold_missing():
    df = _fold_df(4)
    with pytest.raises(AssertionError):
        ensemble_fold_predictions(df, required_folds=5)


def test_best_fold_guardrail():
    df = _fold_df(5)
    df["selected_fold_only"] = False
    assert_no_best_fold_selection(df)
    df.loc[0, "selected_fold_only"] = True
    with pytest.raises(AssertionError):
        assert_no_best_fold_selection(df)
