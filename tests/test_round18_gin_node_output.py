import torch
from torch_geometric.data import Batch, Data

from drugmodels.ginconv import GINConvNet


def _batch(n_graphs=2):
    graphs = []
    for n in (5, 7):
        x = torch.randn(n, 78)
        src = torch.arange(0, n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def test_legacy_forward_shape_unchanged():
    gin = GINConvNet(input_dim=78, output_dim=32, num_layers=5, jk_mode="last", pool_type="max", dropout=0.1)
    out = gin(_batch())
    assert out.shape == (2, 32)


def test_return_node_embeddings_dict():
    gin = GINConvNet(input_dim=78, output_dim=32, num_layers=5, jk_mode="last", pool_type="max", dropout=0.1)
    batch = _batch()
    out = gin(batch, return_node_embeddings=True)
    assert out["node_embeddings"].shape[0] == batch.num_nodes
    assert out["graph_embedding"].shape == (2, 32)
    assert out["batch_index"].numel() == batch.num_nodes


def test_train_mode_default():
    gin = GINConvNet(input_dim=78, output_dim=32)
    gin.train()
    assert gin.training
    for p in gin.parameters():
        assert p.requires_grad
