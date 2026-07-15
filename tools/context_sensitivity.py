#!/usr/bin/env python3
"""Case-level context sensitivity summaries and bootstrap intervals."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def case_bootstrap_delta(
    case_ids: Sequence[object],
    baseline: Sequence[float],
    perturbed: Sequence[float],
    *,
    repeats: int = 2000,
    seed: int = 19,
) -> dict[str, float]:
    case_ids = np.asarray(case_ids)
    delta = np.asarray(perturbed, dtype=float) - np.asarray(baseline, dtype=float)
    if len(case_ids) != len(delta) or not len(delta):
        raise ValueError("aligned non-empty case arrays required")
    unique = np.asarray(sorted(set(case_ids), key=str), dtype=object)
    per_case = np.asarray([delta[case_ids == case].mean() for case in unique])
    rng = np.random.RandomState(seed)
    draws = np.asarray([
        per_case[rng.randint(0, len(per_case), len(per_case))].mean()
        for _ in range(int(repeats))
    ])
    return {
        "n_cases": int(len(unique)),
        "mean_delta": float(per_case.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_unit": "case",
    }
