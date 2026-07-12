import torch
from torch_geometric.data import Batch, Data

from drugmodels.ginconv import GINConvNet
from tools.round18_fusion_models import build_fusion_model


def _gin_batch():
    graphs = []
    for n in (4, 5, 6):
        x = torch.randn(n, 78)
        src = torch.arange(0, n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def test_all_fusion_families_forward():
    gin = GINConvNet(input_dim=78, output_dim=32, num_layers=3, jk_mode="last", pool_type="max")
    batch = _gin_batch()
    gin_out = gin(batch, return_node_embeddings=True)
    omics = torch.randn(3, 40)

    mlp = build_fusion_model("pooled_mlp", omics_dim=40, graph_dim=32)
    assert mlp(omics, gin_out["graph_embedding"]).shape == (3,)

    tf = build_fusion_model(
        "pooled_transformer",
        omics_dim=40,
        graph_dim=32,
        transformer_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
    )
    assert tf(omics, gin_out["graph_embedding"]).shape == (3,)

    pure = build_fusion_model(
        "cross_attention",
        omics_dim=40,
        residual_mode="pure",
        cross_attn_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
    )
    assert pure(omics, gin_out["node_embeddings"], gin_out["batch_index"]).shape == (3,)

    residual = build_fusion_model(
        "cross_attention",
        omics_dim=40,
        residual_mode="pooled_residual",
        cross_attn_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
    )
    assert residual(
        omics,
        gin_out["node_embeddings"],
        gin_out["batch_index"],
        graph_embedding=gin_out["graph_embedding"],
    ).shape == (3,)
