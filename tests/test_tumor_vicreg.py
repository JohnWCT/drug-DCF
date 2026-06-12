import torch
from tools.tumor_vicreg import compute_vicreg_var_cov_loss

def test_vicreg_returns_plain_metric_numbers():
    z = torch.randn(16, 8)
    var_l, cov_l, metrics = compute_vicreg_var_cov_loss(z)
    assert var_l.item() >= 0.0
    assert metrics["tumor_vicreg_valid"] is True

def test_vicreg_small_batch_invalid():
    _, _, metrics = compute_vicreg_var_cov_loss(torch.randn(1, 4))
    assert metrics["tumor_vicreg_valid"] is False
