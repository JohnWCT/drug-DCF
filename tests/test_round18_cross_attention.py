import torch

from tools.cross_attention_switch import CrossAttentionSwitch


def test_cross_attention_shapes_and_padding():
    model = CrossAttentionSwitch(d_model=64, n_heads=4, num_layers=2, dim_feedforward=128)
    model.eval()
    q = torch.randn(4, 1, 64)
    kv = torch.randn(4, 10, 64)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, 7:] = True
    updated, attn = model(q, kv, key_padding_mask=mask, return_attention=True)
    assert updated.shape == (4, 1, 64)
    assert attn.shape == (2, 4, 4, 1, 10)
    assert attn[..., 7:].abs().max().item() < 1e-5
    sums = attn[..., :7].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_permutation_consistency_on_atoms():
    torch.manual_seed(0)
    model = CrossAttentionSwitch(d_model=32, n_heads=4, num_layers=1, dim_feedforward=64)
    model.eval()
    q = torch.randn(1, 1, 32)
    kv = torch.randn(1, 5, 32)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    out1, attn1 = model(q, kv, key_padding_mask=mask, return_attention=True)
    perm = torch.tensor([2, 0, 4, 1, 3])
    kv2 = kv[:, perm, :]
    out2, attn2 = model(q, kv2, key_padding_mask=mask, return_attention=True)
    inv = torch.argsort(perm)
    assert torch.allclose(out1, out2, atol=1e-5)
    assert torch.allclose(attn1, attn2[..., inv], atol=1e-5)
