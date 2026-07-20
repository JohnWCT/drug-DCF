import torch

def test_atom_attention_ndim_and_heads(xa_model, sample_batch, xa_config):
    out = xa_model(*sample_batch, output_mode="attention")
    assert out.atom_attention.ndim == 3
    assert out.atom_attention.shape[1] == xa_config["model"]["cross_attention"]["num_heads"]

def test_valid_atom_normalization(xa_model, sample_batch):
    out = xa_model(*sample_batch, output_mode="attention")
    valid = out.atom_attention * out.atom_mask.unsqueeze(1)
    torch.testing.assert_close(valid.sum(-1), torch.ones_like(valid.sum(-1)), atol=1e-6, rtol=1e-6)

def test_padding_zero(xa_model, sample_batch):
    out = xa_model(*sample_batch, output_mode="attention")
    pad = out.atom_attention.masked_select(~out.atom_mask.unsqueeze(1))
    assert torch.all(pad == 0)
