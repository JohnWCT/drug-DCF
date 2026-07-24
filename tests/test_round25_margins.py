"""Unit tests for Round 25 margin / AADA contracts."""

from __future__ import annotations

import torch

from biocda.losses.prototype_band import prototype_distance_band_loss
from biocda.losses.prototype_margin import margin_gated_prototype_loss
from biocda.losses.smooth_l1_vector import vector_smooth_l1
from biocda.prototypes.margin_estimator import SourceRadiusMarginEstimator
from biocda.stage2.latent_autoencoder import LatentAutoencoder
from biocda.stage2.target_adapter import TargetResidualAdapter


def test_source_anchor_detached():
    t = torch.randn(3, 8, requires_grad=True)
    s = torch.randn(3, 8, requires_grad=True)
    m = torch.zeros(3)
    out = margin_gated_prototype_loss(t, s, m)
    out.loss.backward()
    assert t.grad is not None
    assert s.grad is None


def test_margin_shape():
    t = torch.randn(2, 4)
    s = torch.randn(2, 4)
    try:
        margin_gated_prototype_loss(t, s, torch.zeros(3))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_margin_gate_zero_inside_radius():
    x = torch.randn(2, 16)
    m = torch.full((2,), 0.1)
    out = margin_gated_prototype_loss(x, x.detach(), m)
    assert float(out.loss.item()) == 0.0
    assert float(out.active_fraction.item()) == 0.0


def test_margin_gate_positive_outside_radius():
    a = torch.zeros(1, 4)
    b = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    m = torch.zeros(1)
    out = margin_gated_prototype_loss(a, b, m)
    assert float(out.loss.item()) > 0.0
    assert float(out.active_fraction.item()) == 1.0


def test_band_loss_lower_weight_below_one():
    d = torch.tensor([0.0, 0.5, 1.0])
    lo = torch.full((3,), 0.2)
    hi = torch.full((3,), 0.8)
    out = prototype_distance_band_loss(d, lo, hi, lower_weight=0.1)
    assert float(out.loss.item()) >= 0.0
    try:
        prototype_distance_band_loss(d, lo, hi, lower_weight=1.0)
        assert False
    except ValueError:
        pass


def test_margin_artifact_frozen(tmp_path=None):
    from pathlib import Path
    import tempfile
    est = SourceRadiusMarginEstimator(2, upper_percentile=90, minimum_cancer_observations=2)
    z = torch.randn(8, 4)
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    anchors = torch.randn(2, 4)
    init = torch.ones(2, dtype=torch.bool)
    for _ in range(5):
        est.observe_batch(z, ids, anchors, init, min_count=2)
    art = est.freeze()
    assert art.frozen
    d = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    digest = art.save(d / "margins.json")
    assert len(digest) == 64
    try:
        est.observe_batch(z, ids, anchors, init)
        assert False, "must not observe after freeze"
    except RuntimeError:
        pass


def test_target_adapter_zero_initialized():
    ad = TargetResidualAdapter(16)
    x = torch.randn(5, 16)
    y = ad(x)
    assert torch.allclose(x, y)


def test_smooth_l1_reduces_over_dimensions():
    a = torch.randn(7, 64)
    b = torch.randn(7, 64)
    out = vector_smooth_l1(a, b)
    assert out.shape == (7,)


def test_ae_shape_64():
    ae = LatentAutoencoder(64)
    x = torch.randn(3, 64)
    y = ae(x)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_source_anchor_detached()
    test_margin_shape()
    test_margin_gate_zero_inside_radius()
    test_margin_gate_positive_outside_radius()
    test_band_loss_lower_weight_below_one()
    test_margin_artifact_frozen()
    test_target_adapter_zero_initialized()
    test_smooth_l1_reduces_over_dimensions()
    test_ae_shape_64()
    print("all round25 margin tests passed")
