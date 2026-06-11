import numpy as np

from tools.proto_structure_metrics import compute_proto_structure_metrics


def test_structure_metrics_finite():
    rng = np.random.default_rng(0)
    z_s = rng.normal(size=(20, 8))
    y_s = np.array([0] * 10 + [1] * 10)
    z_t = z_s + 0.1
    y_t = y_s.copy()
    out = compute_proto_structure_metrics(z_s, y_s, z_t, y_t, num_classes=2, kmeans_ari=0.7, kmeans_silhouette=0.5)
    assert out["classwise_domain_gap_mean"] >= 0
    assert 0 <= out["structure_retention_score"] <= 1
