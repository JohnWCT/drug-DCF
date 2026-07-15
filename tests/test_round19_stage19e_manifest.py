"""19E manifest job-count / pin tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path("result/optimization_runs/round19_factorial")


@pytest.mark.integration
def test_stage19e_manifests_exist_and_count():
    lock_path = ROOT / "reports" / "round19_stage19e_candidate_lock.json"
    if not lock_path.is_file():
        pytest.skip("19E candidate lock missing; run setup first")
    lock = json.loads(lock_path.read_text())
    n_cand = len(lock["candidates"])
    meta = json.loads((ROOT / "splits" / "round19e_split_metadata.json").read_text())
    cancer_folds = int(meta.get("cancer_type_n_folds", 5))
    for strategy, folds in [
        ("drug_heldout", 5),
        ("scaffold_heldout", 5),
        ("cancer_type_heldout", cancer_folds),
    ]:
        man = ROOT / "manifests" / f"stage19e_{strategy}_manifest.csv"
        assert man.is_file(), strategy
        df = pd.read_csv(man)
        assert len(df) == n_cand * folds
        assert set(df.candidate_id) == {c["candidate_id"] for c in lock["candidates"]}
        assert not any("tcga" in c.lower() for c in df.columns)
