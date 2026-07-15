from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from tools.round19_attention_consistency import (
    attention_entropy,
    jensen_shannon_divergence,
    spearman_attention,
    topk_overlap,
)
from tools.round19_attention_ensemble import ensemble_atom_attention
from tools.round19_attention_export import (
    export_attention_batches,
    strict_load_locked_models,
)
from tools.round19_context_attention_sensitivity import (
    build_partition_context_controls,
    context_condition_vector,
)
from tools.round19_fusion_models import (
    AdapterMLPFusion,
    AtomCrossAttentionFusionR19,
    PooledTransformerFusionR19,
)
from tools.round19_graph_features import build_pyg_data, legacy_graph_metadata
from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS


def test_opt_in_attention_preserves_default_forward_and_state_dict() -> None:
    torch.manual_seed(19)
    fusion = AtomCrossAttentionFusionR19(
        omics_dim=8, node_dim=6, d_model=8, n_heads=2, num_layers=2,
        dim_feedforward=16, dropout=0.0,
    ).eval()
    keys_before = tuple(fusion.state_dict())
    omics = torch.randn(2, 8)
    nodes = torch.randn(5, 6)
    batch_index = torch.tensor([0, 0, 0, 1, 1])
    default = fusion(omics, nodes, batch_index)
    attention_tuple = fusion(omics, nodes, batch_index, return_attention=True)
    interpreted = fusion(
        omics, nodes, batch_index, return_interpretability=True
    )
    assert torch.equal(default, interpreted["representation"])
    assert torch.equal(default, attention_tuple[0])
    assert torch.equal(attention_tuple[1], interpreted["attention_raw"])
    assert tuple(fusion.state_dict()) == keys_before
    raw = interpreted["attention_raw"]
    assert raw.shape == (2, 2, 2, 1, 3)
    assert torch.equal(raw[:, 1, :, :, 2], torch.zeros_like(raw[:, 1, :, :, 2]))
    assert torch.allclose(raw.sum(-1), torch.ones_like(raw.sum(-1)), atol=1e-6)
    assert torch.allclose(
        interpreted["attention_primary"], raw[-1].mean(1).squeeze(1)
    )


@pytest.mark.parametrize(
    "fusion,args",
    [
        (AdapterMLPFusion(4, 3), (torch.randn(2, 4), torch.randn(2, 3))),
        (PooledTransformerFusionR19(4, 3), (torch.randn(2, 4), torch.randn(2, 3))),
    ],
)
def test_p0_p1_fail_instead_of_returning_fake_atom_attention(fusion, args) -> None:
    with pytest.raises(ValueError, match="no atom-level attention"):
        fusion(*args, return_interpretability=True)


@pytest.mark.parametrize(
    "smiles",
    [
        "C1CCCCC1",             # ring
        "c1ccccc1",             # aromatic
        "CC(=O)[O-].[Na+]",     # salt + charged
        "CCCO",                 # acyclic
        "[NH3+]CC(=O)[O-]",     # charged
    ],
)
def test_gin_gine_atom_mapping_matches_legacy_order(smiles: str) -> None:
    gin = build_pyg_data(smiles, with_bonds=False)
    gine = build_pyg_data(smiles, with_bonds=True)
    meta = legacy_graph_metadata(smiles)
    assert gin.graph_smiles == gine.graph_smiles == meta["graph_smiles"]
    assert torch.equal(gin.x, gine.x)
    assert torch.equal(gin.edge_index, gine.edge_index)
    assert len(meta["atom_metadata"]) == gin.num_nodes
    assert [a["graph_atom_index"] for a in meta["atom_metadata"]] == list(
        range(gin.num_nodes)
    )
    if "." in smiles:
        assert meta["desalt_applied"] is True
        assert len(meta["selected_original_atom_indices"]) == gin.num_nodes


def test_graph_metadata_survives_batch_without_changing_order() -> None:
    graphs = [build_pyg_data("CCO", with_bonds=False), build_pyg_data("c1ccccc1", with_bonds=False)]
    batch = Batch.from_data_list(graphs)
    assert batch.num_graphs == 2
    assert graphs[0].graph_metadata["atom_metadata"][2]["symbol"] == "O"


def test_strict_legacy_checkpoint_load_keeps_keys(tmp_path: Path) -> None:
    modules = (nn.Linear(3, 2), nn.Linear(2, 2), nn.Linear(2, 1))
    checkpoint = {
        "encoder": modules[0].state_dict(),
        "fusion": modules[1].state_dict(),
        "head": modules[2].state_dict(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "member_id": "seed52_fold0",
    }

    def factory(_):
        return nn.Linear(3, 2), nn.Linear(2, 2), nn.Linear(2, 1)

    loaded = strict_load_locked_models(
        record, project_root=tmp_path, model_factory=factory
    )
    assert tuple(loaded[0].state_dict()) == tuple(modules[0].state_dict())


def test_export_streams_same_forward_logit_and_raw_attention(tmp_path: Path) -> None:
    class IdentityNodeEncoder(nn.Module):
        def forward(self, data, return_dict=False, return_graph_embedding=False):
            assert return_dict and not return_graph_embedding
            return {"node_embeddings": data.x[:, :6], "batch_index": data.batch}

    graphs = [
        build_pyg_data("CCO", with_bonds=False),
        build_pyg_data("CC", with_bonds=False),
    ]
    drug_batch = Batch.from_data_list(graphs)
    fusion = AtomCrossAttentionFusionR19(
        omics_dim=8, node_dim=6, d_model=8, n_heads=2, num_layers=1,
        dim_feedforward=16, dropout=0.0,
    )
    head = nn.Linear(8, 1)
    batch = {
        "omics": torch.randn(2, 8),
        "drug_batch": drug_batch,
        "maccs": None,
        "eval_row_id": ["r1", "r2"],
        "ModelID": ["m1", "m2"],
        "drug_name": ["d1", "d2"],
        "target_key": ["internal", "internal"],
        "graph_smiles": [graph.graph_smiles for graph in graphs],
        "legacy_input_smiles": [graph.legacy_input_smiles for graph in graphs],
        "actual_smiles_source": ["lookup", "lookup"],
        "graph_metadata": [graph.graph_metadata for graph in graphs],
    }
    case_shard = pd.DataFrame(
        {
            "eval_row_id": ["r1", "r2"],
            "ModelID": ["m1", "m2"],
            "DRUG_NAME": ["d1", "d2"],
        }
    )
    output = tmp_path / "attention.csv"
    written = export_attention_batches(
        encoder=IdentityNodeEncoder(),
        fusion=fusion,
        head=head,
        dataloader=[batch],
        output_path=output,
        device=torch.device("cpu"),
        encoder_type="gin",
        predictor_id="P2",
        provenance={
            "candidate_id": "C",
            "member_id": "seed52_fold0",
            "checkpoint_path": "checkpoint.pt",
            "checkpoint_sha256": "abc",
            "lock_payload_sha256": "def",
        },
        case_shard=case_shard,
    )
    exported = pd.read_csv(output)
    assert written == len(exported) == 15  # five atoms × (primary + 2 raw heads)
    assert set(exported["attention_kind"]) == {"primary", "raw"}
    assert exported.groupby("eval_row_id")["logit"].nunique().eq(1).all()
    primary = exported[exported["attention_kind"] == "primary"]
    assert np.allclose(primary.groupby("eval_row_id")["attention"].sum(), 1.0)


def test_complete_15_member_attention_ensemble_and_metrics() -> None:
    rows = []
    for member in REQUIRED_MEMBER_IDS:
        for atom, value in enumerate((0.1, 0.2, 0.7)):
            rows.append(
                {
                    "candidate_id": "C",
                    "eval_row_id": "r1",
                    "member_id": member,
                    "atom_index": atom,
                    "attention": value,
                    "is_valid_atom": True,
                    "graph_smiles": "CCO",
                    "ModelID": "M",
                    "drug_name": "D",
                    "target_key": "internal",
                }
            )
    result = ensemble_atom_attention(pd.DataFrame(rows))
    assert set(result["n_members"]) == {15}
    assert np.allclose(result["attention"], [0.1, 0.2, 0.7])
    assert spearman_attention([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
    assert topk_overlap([1, 2, 3], [1, 2, 3], 2) == 1
    assert jensen_shannon_divergence([1, 2], [1, 2]) == pytest.approx(0)
    assert attention_entropy([0, 1]) == pytest.approx(0)


def test_context_controls_are_seeded_and_partition_local() -> None:
    partitions = {"train": ["a", "b"], "test": ["c", "d"]}
    first = build_partition_context_controls(partitions, seed=123)
    assert first == build_partition_context_controls(partitions, seed=123)
    assert set(first["train"].values()) == {"a", "b"}
    assert set(first["test"].values()) == {"c", "d"}
    latent = {
        "a": np.arange(80, dtype=np.float32),
        "b": np.arange(80, dtype=np.float32) + 100,
    }
    shuffled = context_condition_vector(
        latent["a"], condition="shuffled", omics_id="O2", model_id="a",
        latent_by_id=latent, partition_permutation=first["train"],
    )
    zero = context_condition_vector(
        latent["a"], condition="zero", omics_id="O2", model_id="a",
        latent_by_id=latent,
    )
    assert np.array_equal(shuffled[64:80], latent["b"][64:80])
    assert np.count_nonzero(zero[64:80]) == 0
