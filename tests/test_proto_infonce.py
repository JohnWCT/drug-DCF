import torch
import pytest

from tools.proto_infonce import REQUIRED_METRIC_KEYS, compute_batch_prototype_infonce, default_proto_metrics


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
        compute_batch_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=2, temperature=0.0)


def test_loss_is_differentiable_when_valid():
    z_s = torch.randn(4, 6, requires_grad=True)
    y_s = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    z_t = torch.randn(4, 6, requires_grad=True)
    y_t = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss, _ = compute_batch_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=3, temperature=0.2)
    loss.backward()
    assert z_s.grad is not None
    assert z_t.grad is not None


def test_nan_input_raises():
    z_s, y_s, z_t, y_t = _make_batch(num_classes=2, per_class=2)
    z_s[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        compute_batch_prototype_infonce(z_s, y_s, z_t, y_t, num_classes=2, temperature=0.2)


def test_default_metrics_keys():
    metrics = default_proto_metrics()
    for key in REQUIRED_METRIC_KEYS:
        assert key in metrics
