from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from tools.analyze_round19_stage19g import OUTPUT_CSVS, validate_complete
from tools.bond_occlusion import bond_feature_zero, canonical_bond_id
from tools.connected_substructure_masking import connected_mask, matched_connected_random
from tools.context_sensitivity import case_bootstrap_delta
from tools.maccs_ablation import ablate_maccs_bits
from tools.omics_group_ablation import ablate_omics_blocks
from tools.pooled_drug_occlusion import pooled_input_occlusion
from tools.round19_atom_occlusion import batched, feature_zero_graph, matched_random_controls, rank_atom_sets
from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS
from tools.round19_stage19g_executor import _task_and_cases
from tools.scaffold_sidechain_ablation import scaffold_sidechain_partition
from tools.stage19g_routing_audit import audit_routing


def toy_graph() -> Data:
    return Data(
        x=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
        ),
    )


def test_known_top_random_exclusion_and_topology():
    graph = toy_graph()
    ranked = rank_atom_sets([0.1, 0.9, 0.2, 0.3, 0.4, -1.0])
    assert ranked["top1"] == [1] and ranked["top3"] == [1, 4, 3]
    perturbed = feature_zero_graph(graph, ranked["top1"])
    assert torch.equal(perturbed.edge_index, graph.edge_index)
    assert perturbed.x.shape == graph.x.shape and perturbed.x[1].eq(0).all()
    assert max(map(len, batched(list(range(300))))) == 128
    with pytest.raises(ValueError):
        list(batched([1], 129))
    metadata = [
        {"element": "C", "degree": 2, "aromatic": False, "ring": False},
        {"element": "C", "degree": 2, "aromatic": False, "ring": False},
        {"element": "N", "degree": 1, "aromatic": False, "ring": False},
    ]
    controls = matched_random_controls([0], metadata, repeats=20)
    assert len(controls) == 20
    assert all(row["target_excluded"] and 0 not in row["atom_indices"] for row in controls)


def test_connected_pooling_maccs_omics_and_bootstrap():
    graph = toy_graph()
    assert connected_mask([0, 0, 10, 9, 0, 0], graph.edge_index, 0.10) == [2]
    assert all(2 not in row["atom_indices"] for row in matched_connected_random([2], graph.edge_index, 6))
    rows = pooled_input_occlusion(lambda x: x.sum(-1), torch.ones(1, 4), [[0, 1]])
    assert rows[0]["has_attention"] is False and rows[0]["prediction_delta"].item() == 2
    assert ablate_maccs_bits(torch.ones(4), [1], drug_role="D4")[1] == 0
    with pytest.raises(ValueError):
        ablate_maccs_bits(torch.ones(4), [1], drug_role="D3")
    values = np.arange(16).reshape(4, 4)
    shuffled = ablate_omics_blocks(
        values, {"latent_block_0": [0, 1]}, omics_role="O2",
        partition_ids=["a", "a", "b", "b"], seed=9,
    )["latent_block_0"]["shuffled"]
    assert set(shuffled[:2, 0]) == set(values[:2, 0])
    assert set(shuffled[2:, 0]) == set(values[2:, 0])
    summary = case_bootstrap_delta(["a", "a", "b"], [0, 0, 0], [1, 3, 2], repeats=50)
    assert summary["bootstrap_unit"] == "case" and summary["mean_delta"] == 2


def test_bond_and_scaffold_interventions_preserve_semantics():
    graph = toy_graph()
    graph.edge_attr = torch.ones(graph.edge_index.shape[1], 3)
    bond = canonical_bond_id(1, 2, graph.x.shape[0])
    perturbed = bond_feature_zero(graph, [bond], encoder_type="gine")
    assert torch.equal(perturbed.edge_index, graph.edge_index)
    assert perturbed.edge_attr[2:4].eq(0).all()
    with pytest.raises(ValueError):
        bond_feature_zero(graph, [bond], encoder_type="gin")
    partition = scaffold_sidechain_partition("Cc1ccccc1")
    assert len(partition["scaffold"]) == 6
    assert len(partition["sidechain"]) == 1


def test_routing_fold_and_full_development_support(tmp_path):
    cases = pd.DataFrame([
        {"case_id": "a", "evaluation_scope": "19E", "fold_id": 0, "drug_id": "d1",
         "scaffold_id": "s1", "cancer_type": "c1"},
        {"case_id": "b", "evaluation_scope": "TCGA", "fold_id": np.nan, "drug_id": "d2",
         "scaffold_id": "s2", "cancer_type": "new"},
    ])
    support = pd.DataFrame([
        {"fold_id": 0, "support_scope": "fold_train", "drug_id": "d1", "scaffold_id": "s1", "cancer_type": "c1"},
        {"fold_id": -1, "support_scope": "full_development", "drug_id": "d2", "scaffold_id": "s2", "cancer_type": "c2"},
    ])

    def router(lock, novelty, *, project_root):
        roles = {"source_like": "source_performance_champion", "unseen_cancer_type": "cancer_shift_specialist"}
        return {"selected_role": roles[novelty], "lock_file_sha256": "abc"}

    audited = audit_routing(cases, support, final_lock=tmp_path/"lock", project_root=tmp_path, router=router)
    assert audited["routing_match"].all()
    assert list(audited["support_basis"]) == ["19E_fold_relative", "TCGA_full_development"]
    assert {"seen_drug", "seen_scaffold", "seen_cancer_type"} <= set(audited)


def test_task_case_ranges_are_relative_to_their_cohort():
    cases = pd.DataFrame(
        [
            {"eval_row_id": "p1", "selection_reason": "representative_stratified_random"},
            {"eval_row_id": "t1", "selection_reason": "tcga_exploratory"},
            {"eval_row_id": "p2", "selection_reason": "patient_conditioned"},
            {"eval_row_id": "t2", "selection_reason": "tcga_exploratory"},
        ]
    )
    digest = hashlib.sha256("t1\nt2".encode()).hexdigest()
    verified = {
        "_cases": cases,
        "_manifests": {
            "attention": pd.DataFrame(
                [
                    {
                        "task_id": "tcga-task",
                        "cohort_scope": "tcga_exploratory",
                        "case_start": 0,
                        "case_stop_exclusive": 2,
                        "case_count": 2,
                        "case_ids_sha256": digest,
                    }
                ]
            )
        },
    }
    _, selected = _task_and_cases(verified, "attention", "tcga-task", None)
    assert selected["eval_row_id"].tolist() == ["t1", "t2"]


def test_analyzer_complete_gate(tmp_path):
    final_lock = tmp_path / "final.json"
    final_lock.write_text('{"immutable":true}\n')
    final_hash = hashlib.sha256(final_lock.read_bytes()).hexdigest()
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({"final_lock_sha256": final_hash}) + "\n")
    (tmp_path / "experiment_lock.sha256").write_text(
        hashlib.sha256(experiment.read_bytes()).hexdigest() + "\n"
    )
    rows = [
        {"case_id": "case1", "member_id": member, "control_type": "matched_random", "repeat": repeat}
        for member in REQUIRED_MEMBER_IDS for repeat in range(20)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / OUTPUT_CSVS[0], index=False)
    for filename in OUTPUT_CSVS[1:]:
        frame = pd.DataFrame({"case_id": ["case1"]})
        if filename == "round19g_routing_audit.csv":
            frame = pd.DataFrame({
                "case_id": ["case1"], "seen_drug": [True], "seen_scaffold": [True],
                "seen_cancer_type": [True], "routing_match": [True],
            })
        frame.to_csv(tmp_path / filename, index=False)
    result = validate_complete(
        tmp_path, expected_case_ids={"case1"}, final_lock=final_lock, experiment_lock=experiment
    )
    assert result["complete"] is True
