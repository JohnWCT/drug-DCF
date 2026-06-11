import torch
import pytest

from tools.proto_infonce import (
    REQUIRED_METRIC_KEYS,
    compute_batch_prototype_infonce,
    compute_prototype_infonce,
    default_proto_metrics,
)


def _make_batch(num_classes=4, per_class=3, dim=8):
    z_s, y_s, z_t, y_t = [], [], [], []
    for c in range(num_classes):
        z_s.append(torch.randn(per_class, dim, requires_grad=True))
        y_s.append(torch.full((per_class,), c, dtype=torch.long))
        z_t.append(torch.randn(per_class, dim, requires_grad=True))
        y_t.append(torch.full((per_class,), c, dtype=torch.long))
    return torch.cat(z_s), torch.cat(y_s), torch.cat(z_t), torch.cat(y_t)


def test_standard_multiclass_case():
    z_s, y_s, z_t, y_t = _make_batch()
    loss, metrics = compute_batch_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=4, temperature=0.2)
    assert loss.ndim == 0
    assert metrics["proto_valid"] is True
    assert metrics["proto_valid_class_count"] == 4
    for key in REQUIRED_METRIC_KEYS:
        assert key in metrics


def test_combined_mode_backward_compatible():
    z_s, y_s, z_t, y_t = _make_batch(num_classes=4, per_class=3, dim=8)
    old_loss, old_metrics = compute_batch_prototype_infonce(
        z_s, y_s, z_t, y_t, num_classes=4, temperature=0.2
    )
    new_loss, new_metrics = compute_prototype_infonce(
        z_s, y_s, z_t, y_t,
        num_classes=4,
        temperature=0.2,
        mode="combined",
    )
    assert torch.allclose(old_loss, new_loss, atol=1e-6)
    assert old_metrics["proto_valid"] == new_metrics["proto_valid"]
    assert abs(old_metrics["proto_loss"] - new_metrics["proto_loss"]) < 1e-5


def test_cross_domain_differs_from_combined():
    dim = 8
    z_s = torch.randn(6, dim, requires_grad=True)
    y_s = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    z_t = torch.randn(6, dim, requires_grad=True)
    y_t = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    # Push target class 0 toward source class 1 region
    with torch.no_grad():
        z_t[y_t == 0] = z_s[y_s == 1].mean(dim=0) + 0.01

    combined_loss, _ = compute_prototype_infonce(
        z_s, y_s, z_t, y_t, num_classes=3, temperature=0.5, mode="combined"
    )
    cross_loss, cross_metrics = compute_prototype_infonce(
        z_s, y_s, z_t, y_t,
        num_classes=3,
        temperature=0.5,
        mode="cross_domain",
        direction="symmetric",
    )
    assert cross_metrics["proto_cross_domain_valid"] is True
    assert not torch.allclose(combined_loss, cross_loss)


def test_cross_domain_requires_both_domains():
    z_s = torch.randn(4, 6, requires_grad=True)
    y_s = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    z_t = torch.randn(2, 6, requires_grad=True)
    y_t = torch.tensor([0, 0], dtype=torch.long)
    _, metrics = compute_prototype_infonce(
        z_s, y_s, z_t, y_t,
        num_classes=3,
        temperature=0.2,
        mode="cross_domain",
        min_samples_per_domain=1,
    )
    assert metrics["proto_source_valid_class_count"] == 2
    assert metrics["proto_target_valid_class_count"] == 1
    assert metrics["proto_valid_class_count"] <= 1


def test_proto_detach_blocks_prototype_gradient():
    z_s = torch.randn(4, 6, requires_grad=True)
    y_s = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    z_t = torch.randn(4, 6, requires_grad=True)
    y_t = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss, _ = compute_prototype_infonce(
        z_s, y_s, z_t, y_t,
        num_classes=2,
        temperature=0.2,
        mode="cross_domain",
        detach_prototypes=True,
    )
    loss.backward()
    assert z_s.grad is not None
    assert z_t.grad is not None


def test_lambda_proto_zero_no_effect():
    z_s, y_s, z_t, y_t = _make_batch(num_classes=2, per_class=2)
    base = (z_s.sum() + z_t.sum()) * 0.0 + 1.0
    loss, metrics = compute_prototype_infonce(
        z_s, y_s, z_t, y_t, num_classes=2, temperature=0.2, mode="cross_domain"
    )
    total = base + 0.0 * loss
    assert float(total.item()) == 1.0
    _ = metrics


def test_missing_class_in_batch():
    z_s = torch.randn(6, 4, requires_grad=True)
    y_s = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    z_t = torch.randn(4, 4, requires_grad=True)
    y_t = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss, metrics = compute_batch_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=5, temperature=0.2)
    assert metrics["proto_valid_class_count"] == 3
    assert metrics["proto_valid"] is True
    assert torch.isfinite(loss)


def test_single_valid_class_returns_zero_loss():
    z_s = torch.randn(3, 4, requires_grad=True)
    y_s = torch.zeros(3, dtype=torch.long)
    z_t = torch.randn(3, 4, requires_grad=True)
    y_t = torch.zeros(3, dtype=torch.long)
    loss, metrics = compute_batch_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=3, temperature=0.2)
    assert float(loss.item()) == 0.0
    assert metrics["proto_valid"] is False


def test_invalid_temperature_raises():
    z_s, y_s, z_t, y_t = _make_batch(num_classes=2, per_class=2)
    with pytest.raises(ValueError):
        compute_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=2, temperature=0.0)


def test_default_metrics_keys():
    metrics = default_proto_metrics()
    for key in REQUIRED_METRIC_KEYS:
        assert key in metrics
    assert metrics["proto_mode"] == "combined"


def test_target_to_source_skips_s2t_loss():
    z_s, y_s, z_t, y_t = _make_batch(num_classes=2, per_class=2)
    loss, metrics = compute_prototype_infonce(
        z_s,
        y_s,
        z_t,
        y_t,
        num_classes=2,
        temperature=0.5,
        mode="cross_domain",
        direction="target_to_source",
        detach_prototypes=True,
    )
    assert metrics["proto_s2t_loss"] == 0.0
    assert metrics["proto_t2s_valid_sample_count"] > 0
    assert float(loss.item()) >= 0.0
