"""Round 19 fusion compatibility / adapter dim tests."""
import torch

from tools.round19_fusion_models import build_predictor, build_predictor_and_head


def test_p0_adapters_output_128():
    fusion = build_predictor("P0", omics_dim=75, drug_dim=32)
    omics = torch.randn(2, 75)
    drug = torch.randn(2, 32)
    out = fusion(omics, drug)
    assert out.shape == (2, 128)
    assert fusion.output_dim == 128


def test_p1_two_token_maccs_and_graph_drug_dims():
    for drug_dim in (32, 64):
        fusion = build_predictor("P1", omics_dim=91, drug_dim=drug_dim)
        out = fusion(torch.randn(2, 91), torch.randn(2, drug_dim))
        assert out.shape == (2, 64)


def test_p2_accepts_node_dims():
    fusion, head = build_predictor_and_head("P2", omics_dim=91, drug_dim=0, node_dim=64)
    nodes = torch.randn(9, 64)
    batch_index = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
    omics = torch.randn(2, 91)
    repr_vec = fusion(omics, nodes, batch_index)
    logits = head(repr_vec)
    assert logits.shape[0] == 2
    assert torch.isfinite(logits).all()
