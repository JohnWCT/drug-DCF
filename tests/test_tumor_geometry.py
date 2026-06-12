import torch

from tools.tumor_geometry import (
    compute_class_prototypes,
    compute_pairwise_distance_matrix,
    compute_tumor_topology_loss,
)


def _make_balanced(num_classes=4, per_class=5, dim=8):
    labels = []
    rows = []
    for c in range(num_classes):
        labels.extend([c] * per_class)
        rows.extend([torch.randn(dim) + c for _ in range(per_class)])
    z = torch.stack(rows)
    y = torch.tensor(labels, dtype=torch.long)
    return z, y


def test_topology_zero_when_less_than_three_valid_classes():
    z, y = _make_balanced(num_classes=2, per_class=4)
    loss, metrics = compute_tumor_topology_loss(z, y, z, y, num_classes=4)
    assert float(loss.item()) == 0.0
    assert metrics["tumor_topology_valid"] is False


def test_topology_loss_lower_when_distance_matrices_match():
    z_s, y_s = _make_balanced()
    z_t = z_s + 0.01 * torch.randn_like(z_s)
    loss_match, _ = compute_tumor_topology_loss(z_s, y_s, z_t, y_s, num_classes=4)
    z_perm = z_t.clone()
    perm = torch.randperm(z_perm.size(0))
    loss_perm, _ = compute_tumor_topology_loss(z_s, y_s, z_perm, y_s[perm], num_classes=4)
    assert loss_match.item() < loss_perm.item()


def test_topology_loss_higher_when_target_topology_permuted():
    z, y = _make_balanced()
    perm = torch.randperm(z.size(0))
    loss, metrics = compute_tumor_topology_loss(z, y, z[perm], y[perm], num_classes=4)
    assert metrics["tumor_topology_valid"] is True
    assert loss.item() > 0.0


def test_topology_source_detached_target_has_gradient():
    z_s, y_s = _make_balanced()
    z_t = z_s.detach().clone().requires_grad_(True)
    loss, _ = compute_tumor_topology_loss(
        z_s, y_s, z_t, y_s, num_classes=4, detach_source=True
    )
    loss.backward()
    assert z_t.grad is not None


def test_topology_metrics_are_plain_numbers():
    z, y = _make_balanced()
    _, metrics = compute_tumor_topology_loss(z, y, z, y, num_classes=4)
    for key, val in metrics.items():
        if key.endswith("_valid") or key.endswith("_detach_source"):
            assert isinstance(val, bool)
        elif isinstance(val, str):
            continue
        else:
            assert isinstance(val, (int, float))


def test_pairwise_distance_matrix_cosine_distance():
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = compute_pairwise_distance_matrix(p, metric="cosine_distance", normalize=False)
    assert d.shape == (2, 2)
    assert abs(float(d[0, 1].item()) - 1.0) < 1e-5


def test_pairwise_distance_matrix_euclidean():
    p = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    d = compute_pairwise_distance_matrix(p, metric="euclidean", normalize=False)
    assert abs(float(d[0, 1].item()) - 5.0) < 1e-4


def test_compute_class_prototypes():
    z = torch.randn(10, 4)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    p, valid, counts = compute_class_prototypes(z, y, 3, min_samples_per_class=3)
    assert p.shape[0] == 3
    assert len(valid) == 3
