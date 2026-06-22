"""Unit tests for reconstruction loss wiring in pretrain."""

from tools.reconstruction_losses import reconstruction_loss_kwargs
from tools.model_opt import vaeloss

import torch


def test_reconstruction_kwargs_unpack_into_vaeloss():
    param = {
        "reconstruction_loss_type": "smooth_l1",
        "smooth_l1_beta": 0.5,
        "reconstruction_loss_scale": 2.0,
    }
    kw = reconstruction_loss_kwargs(param)
    x = torch.randn(4, 8)
    mu = torch.zeros(4, 8)
    sigma = torch.zeros(4, 8)
    recon = x + 0.1 * torch.randn(4, 8)
    loss = vaeloss(mu, sigma, recon, x, **kw)
    assert torch.isfinite(loss)
    assert kw["reconstruction_loss_type"] == "smooth_l1"
