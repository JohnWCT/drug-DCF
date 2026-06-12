import torch
from tools.tumor_supcon import compute_within_domain_supcon_loss

def test_supcon_valid_with_enough_samples():
    z = torch.randn(12, 8)
    y = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3])
    loss, metrics = compute_within_domain_supcon_loss(z, y, z, y, min_samples_per_class=2)
    assert metrics["tumor_supcon_valid"] is True
    assert float(loss.item()) >= 0.0
