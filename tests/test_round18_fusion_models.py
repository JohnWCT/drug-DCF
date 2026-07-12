import torch
from torch_geometric.data import Batch, Data

from drugmodels.ginconv import GINConvNet
from tools.round18_fusion_models import build_fusion_and_head


def _gin_batch():
    graphs = []
    for n in (4, 5, 6):
        x = torch.randn(n, 78)
        src = torch.arange(0, n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def test_all_fusion_families_forward_with_separate_head():
    gin = GINConvNet(input_dim=78, output_dim=32, num_layers=3, jk_mode="last", pool_type="max")
    batch = _gin_batch()
    gin_out = gin(batch, return_node_embeddings=True)
    omics = torch.randn(3, 40)

    fusion, head = build_fusion_and_head("pooled_mlp", omics_dim=40, graph_dim=32)
    logits = head(fusion(omics, gin_out["graph_embedding"]))
    assert logits.shape == (3,)
    # no parameter overlap between fusion and head
    fusion_ids = {id(p) for p in fusion.parameters()}
    head_ids = {id(p) for p in head.parameters()}
    assert fusion_ids.isdisjoint(head_ids)

    fusion, head = build_fusion_and_head(
        "pooled_transformer",
        omics_dim=40,
        graph_dim=32,
        transformer_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128, "use_mask": True},
    )
    assert fusion.metadata()["effective_use_mask"] is False
    assert head(fusion(omics, gin_out["graph_embedding"])).shape == (3,)

    fusion, head = build_fusion_and_head(
        "cross_attention",
        omics_dim=40,
        residual_mode="pure",
        cross_attn_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
    )
    assert head(fusion(omics, gin_out["node_embeddings"], gin_out["batch_index"])).shape == (3,)

    fusion, head = build_fusion_and_head(
        "cross_attention",
        omics_dim=40,
        residual_mode="pooled_residual",
        cross_attn_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
    )
    assert head(
        fusion(
            omics,
            gin_out["node_embeddings"],
            gin_out["batch_index"],
            graph_embedding=gin_out["graph_embedding"],
        )
    ).shape == (3,)
