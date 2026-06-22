import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tools.round9_diagnostics_common import fit_domain_classifier, leakage_strength

def test_fit_domain_classifier_logistic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = np.array([0] * 20 + [1] * 20)
    auc, bal = fit_domain_classifier(x, y, "logistic_regression", random_state=0)
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= bal <= 1.0

def test_leakage_strength_for_reversed_auc():
    assert leakage_strength(0.2) == 0.3

def test_insufficient_samples_not_in_macro(tmp_path):
    from tools.round9_diagnostics_common import macro_mean
    vals = [0.6, float("nan")]
    assert macro_mean(vals) == 0.6
