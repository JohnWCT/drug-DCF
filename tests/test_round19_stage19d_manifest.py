"""Tests for Round 19D manifest generation."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import pytest
from tools.round19_config_builder import build_stage19d_manifest, _load_json

ROOT = Path("result/optimization_runs/round19_factorial")
SETTINGS = Path("config/round19_factorial_settings.json")

@pytest.mark.integration
def test_stage19d_manifest_counts_and_seeds(tmp_path):
    prop_path = ROOT / "reports" / "round19_stage19d_candidate_proposal.json"
    if not prop_path.is_file():
        pytest.skip("proposal missing")
    settings = _load_json(SETTINGS)
    # Use real outdir so splits land in expected place; regenerate is idempotent
    proposal = json.loads(prop_path.read_text())
    df = build_stage19d_manifest(settings, str(ROOT), proposal, split_seeds=[52,62,72], n_folds=5)
    n = int(proposal["n_candidates"])
    assert len(df) == n * 15
    assert set(df.split_seed.astype(int)) == {52,62,72}
    assert set(df.fold_id.astype(int)) == {0,1,2,3,4}
    assert df.job_id.is_unique
    assert df.result_dir.is_unique
    for cid, g in df.groupby("candidate_id"):
        assert len(g) == 15
    assert not df.columns.str.contains("TCGA|Integrated5|internal_test_auc", case=False).any()
    f4 = df[df.candidate_id=="F4_source_only_o4"].iloc[0]
    assert (f4.drug_id, f4.predictor_id, f4.omics_id) == ("D3","P2","O4")
