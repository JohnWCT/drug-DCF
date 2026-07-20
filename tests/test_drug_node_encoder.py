from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.drug_gin import DrugGINNodeEncoder
import torch

def test_drug_encoder_returns_node_embeddings():
    enc = DrugGINNodeEncoder(node_hidden_dim=32)
    out = enc(make_chain_graph(8))
    assert out.node_embeddings.shape[0] == 8

def test_node_embedding_count_matches_total_atoms():
    batch = batch_drug_graphs([make_chain_graph(5), make_chain_graph(9)])
    out = DrugGINNodeEncoder()(batch)
    assert out.node_embeddings.shape[0] == 14

def test_batch_index_matches_total_atoms():
    batch = batch_drug_graphs([make_chain_graph(4), make_chain_graph(7)])
    out = DrugGINNodeEncoder()(batch)
    assert out.batch_index.shape[0] == out.node_embeddings.shape[0]

def test_model_atom_index_is_preserved():
    out = DrugGINNodeEncoder()(make_chain_graph(6))
    assert torch.equal(out.model_atom_index, torch.arange(6))

def test_no_graph_pooling_in_node_encoder():
    out = DrugGINNodeEncoder()(make_chain_graph(3))
    assert out.node_embeddings.shape[0] == 3
