import torch

from tools.classwise_alignment import compute_classwise_mmd


def test_classwise_mmd_zero_when_no_valid_class():
    z_s = torch.randn(2, 4, requires_grad=True)
    y_s = torch.tensor([0, 0], dtype=torch.long)
    z_t = torch.randn(2, 4, requires_grad=True)
    y_t = torch.tensor([1, 1], dtype=torch.long)
    loss, metrics = compute_classwise_mmd(
        z_s, y_s, z_t, y_t, num_classes=2, min_samples_per_domain=2
    )
    assert float(loss.item()) == 0.0
    assert metrics["cmmd_valid"] is False


def test_classwise_mmd_positive_for_shifted_domains():
    z_s = torch.randn(8, 4, requires_grad=True)
    y_s = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    z_t = torch.randn(8, 4, requires_grad=True)
    y_t = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    with torch.no_grad():
        z_t += 5.0
    loss, metrics = compute_classwise_mmd(
        z_s, y_s, z_t, y_t, num_classes=2, min_samples_per_domain=2
    )
    assert metrics["cmmd_valid"] is True
    assert metrics["cmmd_loss"] > 0


def test_classwise_mmd_lower_for_aligned_domains():
    torch.manual_seed(0)
    z_s = torch.randn(8, 4)
    y_s = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    z_t_aligned = z_s.clone()
    z_t_shifted = z_s.clone() + 3.0
    y_t = y_s.clone()
    _, m_aligned = compute_classwise_mmd(z_s, y_s, z_t_aligned, y_t, num_classes=2, min_samples_per_domain=2)
    _, m_shifted = compute_classwise_mmd(z_s, y_s, z_t_shifted, y_t, num_classes=2, min_samples_per_domain=2)
    assert m_aligned["cmmd_loss"] < m_shifted["cmmd_loss"]


def test_cmmd_no_effect_when_lambda_zero():
    z_s = torch.randn(4, 4, requires_grad=True)
    y_s = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    z_t = torch.randn(4, 4, requires_grad=True)
    y_t = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    base = z_s.sum() * 0.0 + 2.0
    cmmd_loss, _ = compute_classwise_mmd(z_s, y_s, z_t, y_t, num_classes=2, min_samples_per_domain=2)
    total = base + 0.0 * cmmd_loss
    assert float(total.item()) == 2.0
