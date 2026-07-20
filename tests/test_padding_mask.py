import torch
from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.cross_attention import SampleAtomCrossAttention
from biocda.models.drug_gin import DrugGINNodeEncoder

def test_atom_mask_true_for_valid_atoms():
    batch = batch_drug_graphs([make_chain_graph(7), make_chain_graph(3)])
    enc = DrugGINNodeEncoder()
    attn = SampleAtomCrossAttention(sample_dim=96, node_dim=enc.node_dim)
    nodes = enc(batch)
    out = attn(torch.randn(2, 96), nodes.node_embeddings, nodes.batch_index)
    assert out.atom_mask[0, :7].all() and out.atom_mask[1, :3].all()
    assert not out.atom_mask[0, 7:].any()
