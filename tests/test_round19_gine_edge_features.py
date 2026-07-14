"""Round 19 GINE edge-feature tests."""
import torch
from torch_geometric.data import Batch

from drugmodels.gineconv import GINEConvNet
from tools.round19_graph_features import BOND_FEATURE_DIM, build_pyg_data


def test_gine_edge_attr_changes_output():
    smiles = "CCO"
    g0 = build_pyg_data(smiles, with_bonds=True)
    g1 = build_pyg_data(smiles, with_bonds=True)
    assert g0.edge_attr is not None
    assert g0.edge_attr.shape[-1] == BOND_FEATURE_DIM

    model = GINEConvNet(
        input_dim=78,
        edge_dim=BOND_FEATURE_DIM,
        node_hidden_dim=64,
        graph_output_dim=64,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()
    b0 = Batch.from_data_list([g0])
    y0 = model(b0).detach().clone()

    g1.edge_attr = torch.zeros_like(g1.edge_attr)
    b1 = Batch.from_data_list([g1])
    y1 = model(b1).detach()
    assert not torch.allclose(y0, y1, atol=1e-6)


def test_gine_forward_dict_and_grads():
    smiles = ["CCO", "c1ccccc1"]
    batch = Batch.from_data_list([build_pyg_data(s, with_bonds=True) for s in smiles])
    model = GINEConvNet(node_hidden_dim=64, graph_output_dim=64, edge_dim=BOND_FEATURE_DIM, num_layers=2)
    model.train()
    out = model(batch, return_dict=True)
    loss = out["graph_embedding"].pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
