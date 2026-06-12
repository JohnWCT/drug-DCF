import torch
from tools.tumor_subspace import (
    classifier_input_dim, compute_subspace_orthogonality_loss,
    discriminator_input_dim, resolve_subspace_training_params,
    select_latent_view, split_tumor_transfer_latent,
)

def test_split_tumor_transfer_latent():
    z = torch.randn(4, 32)
    z_t, z_tr = split_tumor_transfer_latent(z, 8)
    assert z_t.shape == (4, 8)
    assert z_tr.shape == (4, 24)

def test_subspace_disabled_uses_shared():
    cfg = resolve_subspace_training_params({"latent_size": 32, "use_tumor_subspace": False})
    z = torch.randn(2, 32)
    assert select_latent_view(z, "tumor", cfg).shape == z.shape

def test_classifier_and_discriminator_dims():
    cfg = resolve_subspace_training_params({
        "latent_size": 32, "use_tumor_subspace": True, "tumor_dim": 8,
        "alignment_latent_view": "transfer", "classifier_latent_view": "tumor",
    })
    assert classifier_input_dim(cfg) == 8
    assert discriminator_input_dim(cfg) == 48

def test_subspace_orthogonality_loss():
    loss = compute_subspace_orthogonality_loss(torch.randn(32, 4), torch.randn(32, 8))
    assert loss.item() >= 0.0
