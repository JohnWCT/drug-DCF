import torch
from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.cross_attention import SampleAtomCrossAttention
from biocda.models.drug_gin import DrugGINNodeEncoder

def _run():
    enc = DrugGINNodeEncoder(node_hidden_dim=32)
    attn = SampleAtomCrossAttention(sample_dim=96, node_dim=enc.node_dim, attention_dim=64, num_heads=4)
    batch = batch_drug_graphs([make_chain_graph(10), make_chain_graph(4)])
    nodes = enc(batch)
    return attn(torch.randn(2, 96), nodes.node_embeddings, nodes.batch_index)

def test_attention_shape():
    out = _run()
    assert out.attention_probabilities.shape[0] == 2 and out.attention_probabilities.shape[1] == 4

def test_attention_logits_shape():
    out = _run()
    assert out.attention_logits.shape == out.attention_probabilities.shape

def test_padding_attention_is_zero():
    out = _run()
    pad = out.attention_probabilities.masked_select(~out.atom_mask.unsqueeze(1))
    assert torch.all(pad == 0)

def test_valid_attention_sums_to_one():
    out = _run()
    valid = out.attention_probabilities * out.atom_mask.unsqueeze(1)
    torch.testing.assert_close(valid.sum(-1), torch.ones_like(valid.sum(-1)), atol=1e-6, rtol=1e-6)

def test_attention_has_no_nan():
    assert not torch.isnan(_run().attention_probabilities).any()

def test_attention_has_no_inf():
    assert not torch.isinf(_run().attention_probabilities).any()

def test_single_atom_graph_attention_equals_one():
    enc = DrugGINNodeEncoder()
    enc.eval()
    attn = SampleAtomCrossAttention(sample_dim=96, node_dim=enc.node_dim)
    attn.eval()
    nodes = enc(make_chain_graph(1))
    out = attn(torch.randn(1, 96), nodes.node_embeddings, nodes.batch_index)
    assert torch.allclose(out.attention_probabilities, torch.ones_like(out.attention_probabilities))

def test_different_sample_queries_can_change_attention():
    enc = DrugGINNodeEncoder()
    attn = SampleAtomCrossAttention(sample_dim=96, node_dim=enc.node_dim)
    batch = batch_drug_graphs([make_chain_graph(8)])
    nodes = enc(batch)
    a = attn(torch.randn(1, 96), nodes.node_embeddings, nodes.batch_index)
    b = attn(torch.randn(1, 96), nodes.node_embeddings, nodes.batch_index)
    assert not torch.allclose(a.attention_probabilities, b.attention_probabilities)
