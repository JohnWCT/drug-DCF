"""Tests for Round 19D candidate proposal selector."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from tools.round19_stage19d_selector import build_proposal, maybe_graph_f6, pick_best_pooled_o2

ROOT = Path("result/optimization_runs/round19_factorial")

@pytest.mark.integration
def test_stage19d_proposal_roles_and_counts():
    if not (ROOT / "reports" / "round19c_full_composition_ranking.csv").is_file():
        pytest.skip("19C composition ranking missing")
    prop = build_proposal(ROOT)
    assert prop["lock_type"] == "stage19d_candidate_proposal"
    assert prop["internal_test_used"] is False
    assert prop["tcga_used"] is False
    ids = [c["candidate_id"] for c in prop["candidates"]]
    for need in ["F0_historical_anchor","F1_primary_o2","F2_full_omics_o3","F3_best_pooled_o2","F4_source_only_o4"]:
        assert need in ids
    assert 5 <= len(ids) <= 6
    f4 = next(c for c in prop["candidates"] if c["candidate_id"] == "F4_source_only_o4")
    assert (f4["drug_id"], f4["predictor_id"], f4["omics_id"]) == ("D3", "P2", "O4")

def test_pick_best_pooled_o2_prefers_p0_p1():
    comp = pd.DataFrame([
        {"drug_id":"D0","predictor_id":"P2","omics_id":"O2","mean_drugmacro_auc":0.9,"std_drugmacro_auc":0.01,"mean_drugmacro_auprc":0.5},
        {"drug_id":"D0","predictor_id":"P0","omics_id":"O2","mean_drugmacro_auc":0.8,"std_drugmacro_auc":0.01,"mean_drugmacro_auprc":0.4},
        {"drug_id":"D4","predictor_id":"P1","omics_id":"O2","mean_drugmacro_auc":0.81,"std_drugmacro_auc":0.01,"mean_drugmacro_auprc":0.41},
    ])
    d,p,m = pick_best_pooled_o2(comp)
    assert p in {"P0","P1"}
    assert (d,p)==("D4","P1")

def test_f6_rejects_without_delta():
    comp = pd.DataFrame([
        {"drug_id":"D0","predictor_id":"P2","omics_id":"O2","mean_drugmacro_auc":0.62,"std_drugmacro_auc":0.01,"mean_drugmacro_auprc":0.4},
        {"drug_id":"D2","predictor_id":"P2","omics_id":"O2","mean_drugmacro_auc":0.621,"std_drugmacro_auc":0.01,"mean_drugmacro_auprc":0.4},
        {"drug_id":"D3","predictor_id":"P2","omics_id":"O2","mean_drugmacro_auc":0.619,"std_drugmacro_auc":0.01,"mean_drugmacro_auprc":0.4},
    ])
    assert maybe_graph_f6(comp) is None
