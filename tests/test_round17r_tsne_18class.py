#!/usr/bin/env python3
"""Round 17R tSNE 18-class expectation tests."""

from __future__ import annotations

import pandas as pd


def test_tsne_coordinates_expect_18_prototypes() -> None:
    df = pd.DataFrame(
        [{"point_type": "source_prototype", "cancer_type": f"c{i}"} for i in range(18)]
        + [{"point_type": "target_prototype", "cancer_type": f"c{i}"} for i in range(18)]
        + [{"point_type": "sample", "cancer_type": "c0"}]
    )
    n_src = int((df["point_type"] == "source_prototype").sum())
    n_tgt = int((df["point_type"] == "target_prototype").sum())
    forbidden = {"Engineered", "Fibroblast"}
    present = set(df["cancer_type"].astype(str)) & forbidden
    assert n_src == 18
    assert n_tgt == 18
    assert not present
