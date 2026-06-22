"""Tests for reconstruction loss helpers."""

import torch

from tools.reconstruction_losses import (
    compute_reconstruction_loss,
    reconstruction_loss_defaults,
    reconstruction_loss_kwargs,
)


def test_mse_matches_functional():
    x = torch.randn(4, 8)
    recon = x + 0.1 * torch.randn(4, 8)
    expected = torch.nn.functional.mse_loss(recon, x, reduction="mean")
    got = compute_reconstruction_loss(recon, x, loss_type="mse")
    assert torch.allclose(got, expected)


def test_smooth_l1_matches_functional():
    x = torch.randn(4, 8)
    recon = x + torch.randn(4, 8)
    for beta in (0.25, 0.5, 1.0, 2.0):
        expected = torch.nn.functional.smooth_l1_loss(recon, x, beta=beta, reduction="mean")
        got = compute_reconstruction_loss(recon, x, loss_type="smooth_l1", smooth_l1_beta=beta)
        assert torch.allclose(got, expected)


def test_hybrid_backward():
    x = torch.randn(3, 5, requires_grad=True)
    recon = torch.randn(3, 5, requires_grad=True)
    loss = compute_reconstruction_loss(recon, x, loss_type="hybrid_mse_smooth_l1", hybrid_alpha=0.5)
    loss.backward()
    assert recon.grad is not None


def test_invalid_type_raises():
    x = torch.randn(2, 2)
    try:
        compute_reconstruction_loss(x, x, loss_type="l1_only")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_defaults_preserve_mse():
    defaults = reconstruction_loss_defaults()
    assert defaults["reconstruction_loss_type"] == "mse"
    kw = reconstruction_loss_kwargs({})
    assert kw["reconstruction_loss_type"] == "mse"
