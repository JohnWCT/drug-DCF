
import pandas as pd
import pytest

from tools.round19_stage19f_ensemble import (
    REQUIRED_FOLDS,
    REQUIRED_SEEDS,
    ensemble_predictions,
    validate_ensemble_predictions,
)


def predictions():
    rows = []
    for seed in REQUIRED_SEEDS:
        for fold in REQUIRED_FOLDS:
            rows.append({
                "candidate_id": "F3_best_pooled_o2",
                "eval_row_id": "row-1",
                "split_seed": seed,
                "fold_id": fold,
                "Label": 1,
                "ModelID": "M1",
                "drug_name": "drug-a",
                "target_key": "internal_posthoc",
                "probability": 0.4 + 0.001 * fold,
            })
    return pd.DataFrame(rows)


def test_complete_15_member_mean():
    out = ensemble_predictions(predictions())
    assert len(out) == 1
    assert int(out.iloc[0].n_members) == 15


def test_missing_duplicate_and_identity_drift_fail():
    frame = predictions()
    with pytest.raises(AssertionError):
        validate_ensemble_predictions(frame.iloc[:-1])
    with pytest.raises(AssertionError):
        validate_ensemble_predictions(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    drift = frame.copy()
    drift.loc[0, "Label"] = 0
    with pytest.raises(AssertionError):
        validate_ensemble_predictions(drift)
