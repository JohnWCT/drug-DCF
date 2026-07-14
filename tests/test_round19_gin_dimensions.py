"""Round 19 GIN dimension / legacy compatibility tests."""
import torch
from torch_geometric.data import Batch, Data

from drugmodels.ginconv import GINConvNet


def _toy_batch(n_graphs=2):
    graphs = []
    for n in (4, 5):
        x = torch.randn(n, 78)
        src = torch.arange(0, n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def test_legacy_output_dim_keeps_node32():
    gin = GINConvNet(input_dim=78, output_dim=32)
    assert gin.node_hidden_dim == 32
    assert gin.graph_output_dim == 32
    batch = _toy_batch()
    out = gin(batch)
    assert out.shape == (2, 32)


def test_d0_d1_d2_dims():
    batch = _toy_batch()
    d0 = GINConvNet(node_hidden_dim=32, graph_output_dim=32)
    d1 = GINConvNet(node_hidden_dim=64, graph_output_dim=32)
    d2 = GINConvNet(node_hidden_dim=64, graph_output_dim=64)
    for model, node_d, graph_d in ((d0, 32, 32), (d1, 64, 32), (d2, 64, 64)):
        out = model(batch, return_dict=True)
        assert out["node_embeddings"].shape[-1] == node_d
        assert out["graph_embedding"].shape == (2, graph_d)
        assert out["node_dim"] == node_d
        assert out["graph_dim"] == graph_d
