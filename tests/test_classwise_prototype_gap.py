import torch
from tools.classwise_alignment import compute_classwise_prototype_gap


def _make_aligned(num_classes=3, n=8, dim=4):
    z_s, z_t, y_s, y_t = [], [], [], []
    for c in range(num_classes):
        center = torch.randn(dim) + c * 3.0
        z_s.append(center.unsqueeze(0).expand(n, -1) + 0.01 * torch.randn(n, dim))
        z_t.append(center.unsqueeze(0).expand(n, -1) + 0.01 * torch.randn(n, dim))
        y_s.append(torch.full((n,), c, dtype=torch.long))
        y_t.append(torch.full((n,), c, dtype=torch.long))
    return torch.cat(z_s), torch.cat(y_s), torch.cat(z_t), torch.cat(y_t)


def test_class_gap_zero_when_no_valid_class():
    z = torch.randn(4, 8, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1])
    loss, m = compute_classwise_prototype_gap(z, y, z, y, num_classes=2, min_samples_per_domain=3)
    assert m["class_gap_valid"] is False
    assert float(loss.item()) == 0.0


def test_class_gap_cosine_lower_when_prototypes_aligned():
    z_s, y_s, z_t, y_t = _make_aligned()
    loss_aligned, _ = compute_classwise_prototype_gap(z_s, y_s, z_t, y_t, num_classes=3, metric="cosine")
    z_far = z_t + 5.0
    loss_far, _ = compute_classwise_prototype_gap(z_s, y_s, z_far, y_t, num_classes=3, metric="cosine")
    assert loss_aligned.item() < loss_far.item()


def test_class_gap_l2_lower_when_prototypes_aligned():
    z_s, y_s, z_t, y_t = _make_aligned()
    loss_aligned, m = compute_classwise_prototype_gap(z_s, y_s, z_t, y_t, num_classes=3, metric="l2")
    assert m["class_gap_metric"] == "l2"
    z_far = z_t + 5.0
    loss_far, _ = compute_classwise_prototype_gap(z_s, y_s, z_far, y_t, num_classes=3, metric="l2")
    assert loss_aligned.item() < loss_far.item()


def test_class_gap_source_detached_target_has_gradient():
    z_s = torch.randn(16, 4)
    z_t = torch.randn(16, 4, requires_grad=True)
    y = torch.tensor([0] * 8 + [1] * 8)
    loss, _ = compute_classwise_prototype_gap(
        z_s, y, z_t, y, num_classes=2, metric="cosine", detach_source=True, detach_target=False
    )
    loss.backward()
    assert z_t.grad is not None


def test_class_gap_metrics_are_plain_numbers():
    z_s, y_s, z_t, y_t = _make_aligned()
    _, m = compute_classwise_prototype_gap(z_s, y_s, z_t, y_t, num_classes=3, metric="cosine")
    for k, v in m.items():
        if k.endswith("_detach_source") or k.endswith("_detach_target") or k == "class_gap_valid":
            assert isinstance(v, bool)
        elif k in {"class_gap_metric", "class_gap_l2_squared"}:
            continue
        else:
            assert isinstance(v, (int, float))
