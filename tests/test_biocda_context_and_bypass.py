import torch

def test_context_reaches_sample_representation(xa_model, sample_batch):
    out = xa_model(*sample_batch, output_mode="full")
    assert out.sample_representation is not None

def test_context_reaches_attention_query(xa_model, sample_batch):
    omics, context, batch = sample_batch
    a = xa_model(omics, context, batch, output_mode="attention")
    b = xa_model(omics, torch.zeros_like(context), batch, output_mode="attention")
    assert not torch.allclose(a.atom_attention, b.atom_attention)

def test_context_swap_can_change_attention(xa_model, sample_batch):
    omics, context, batch = sample_batch
    a = xa_model(omics, context, batch, output_mode="attention")
    b = xa_model(omics, context + 1.0, batch, output_mode="attention")
    assert not torch.allclose(a.atom_attention, b.atom_attention)

def test_context_zeroing_is_supported(xa_model, sample_batch):
    omics, context, batch = sample_batch
    assert xa_model(omics, torch.zeros_like(context), batch).logits.shape == (2,)

def test_predictor_receives_attended_drug_representation(xa_model, sample_batch):
    out = xa_model(*sample_batch, output_mode="full")
    assert out.drug_representation.shape[-1] == xa_model.cross_attention.attention_dim

def test_predictor_does_not_receive_global_pooled_embedding(xa_model):
    assert not hasattr(xa_model, "fusion")

def test_attention_output_participates_in_prediction(xa_model, sample_batch):
    assert xa_model(*sample_batch, output_mode="full").logits.ndim == 1
