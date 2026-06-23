"""Tests for source-anchor EMA prototype alignment."""

import torch

from tools.source_anchor_prototypes import (
    resolve_source_anchor_proto_training_params,
    SourceAnchorEMAPrototypes,
    compute_batch_prototypes,
    compute_source_anchor_alignment_loss,
    get_proto_align_lambda_eff,
)


def test_ema_init_shape():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=5, latent_size=8, device="cpu")
    assert ema.prototypes.shape == (5, 8)
    assert ema.initialized_count() == 0


def test_update_no_grad():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=3, latent_size=4, momentum=0.9, device="cpu")
    z = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    ema.update(z, labels, min_count=2)
    assert ema.initialized_count() == 3
    assert ema.prototypes.grad is None


def test_get_stop_gradient():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=2, latent_size=3, device="cpu")
    z = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 1, 1])
    ema.update(z, labels, min_count=2)
    anchors = ema.get(torch.tensor([0, 1]))
    assert not anchors.requires_grad


def test_compute_batch_prototypes_min_count():
    z = torch.randn(5, 4)
    labels = torch.tensor([0, 0, 1, 2, 2])
    protos = compute_batch_prototypes(z, labels, num_cancer_types=3, min_count=2)
    assert 0 in protos and 2 in protos
    assert 1 not in protos


def test_cosine_alignment_backward_to_target():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=2, latent_size=4, device="cpu")
    src = torch.randn(4, 4)
    ema.update(src, torch.tensor([0, 0, 1, 1]), min_count=2)
    target_z = torch.randn(4, 4, requires_grad=True)
    tgt_labels = torch.tensor([0, 0, 1, 1])
    loss, metrics = compute_source_anchor_alignment_loss(
        target_z, tgt_labels, ema, metric="cosine", min_count=2
    )
    assert metrics["proto_align_num_cancers"] == 2
    loss.backward()
    assert target_z.grad is not None
    assert float(target_z.grad.abs().sum()) > 0


def test_source_z_no_alignment_gradient():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=1, latent_size=3, device="cpu")
    source_z = torch.randn(3, 3, requires_grad=True)
    ema.update(source_z, torch.tensor([0, 0, 0]), min_count=2)
    assert source_z.grad is None


def test_uninitialized_cancer_skipped():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=3, latent_size=2, device="cpu")
    ema.update(torch.randn(2, 2), torch.tensor([0, 0]), min_count=2)
    target_z = torch.randn(2, 2)
    loss, metrics = compute_source_anchor_alignment_loss(
        target_z, torch.tensor([1, 1]), ema, min_count=2
    )
    assert metrics["proto_align_num_cancers"] == 0
    assert float(loss.item()) == 0.0


def test_invalid_metric_raises():
    ema = SourceAnchorEMAPrototypes(num_cancer_types=1, latent_size=2, device="cpu")
    try:
        compute_source_anchor_alignment_loss(
            torch.randn(2, 2), torch.tensor([0, 0]), ema, metric="l1"
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_proto_align_lambda_schedule():
    assert get_proto_align_lambda_eff(10, 0.001, 20, 60) == 0.0
    assert get_proto_align_lambda_eff(60, 0.001, 20, 60) == 0.001
    mid = get_proto_align_lambda_eff(40, 0.001, 20, 60)
    assert 0.0 < mid < 0.001


def test_invalid_metric_in_resolve_raises():
    try:
        resolve_source_anchor_proto_training_params(
            {"proto_align_metric": "l1"}
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_momentum_raises():
    try:
        resolve_source_anchor_proto_training_params(
            {"proto_ema_momentum": 1.5}
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_min_count_raises():
    try:
        resolve_source_anchor_proto_training_params(
            {"proto_align_min_count": 0}
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
