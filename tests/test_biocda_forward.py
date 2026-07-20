import pytest
import torch

def test_prediction_output_mode(xa_model, sample_batch):
    omics, context, batch = sample_batch
    out = xa_model(omics, context, batch, output_mode="prediction")
    assert out.logits.shape == (2,) and out.atom_attention is None

def test_attention_output_mode(xa_model, sample_batch, xa_config):
    out = xa_model(*sample_batch, output_mode="attention")
    assert out.atom_attention.shape[1] == xa_config["model"]["cross_attention"]["num_heads"]

def test_full_output_mode(xa_model, sample_batch):
    out = xa_model(*sample_batch, output_mode="full")
    assert out.sample_representation is not None and out.node_embeddings is not None

def test_invalid_output_mode_fails(xa_model, sample_batch):
    with pytest.raises(ValueError):
        xa_model(*sample_batch, output_mode="invalid")

def test_output_mode_does_not_change_logits(xa_model, sample_batch):
    a = xa_model(*sample_batch, output_mode="prediction")
    b = xa_model(*sample_batch, output_mode="full")
    torch.testing.assert_close(a.logits, b.logits)

def test_probabilities_equal_sigmoid_logits(xa_model, sample_batch):
    out = xa_model(*sample_batch, output_mode="prediction")
    torch.testing.assert_close(out.probabilities, torch.sigmoid(out.logits))
