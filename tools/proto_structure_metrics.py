"""Reporting-only prototype / domain structure metrics (Round 4.1)."""

from __future__ import annotations

import numpy as np


def _class_prototypes(z: np.ndarray, y: np.ndarray, num_classes: int, min_count: int):
    protos = {}
    for c in range(int(num_classes)):
        mask = y == c
        if int(mask.sum()) >= min_count:
            protos[c] = z[mask].mean(axis=0)
    return protos


def _inter_class_margin(protos: dict) -> float:
    if len(protos) < 2:
        return 0.0
    keys = sorted(protos.keys())
    dists = []
    for i, ci in enumerate(keys):
        for cj in keys[i + 1 :]:
            dists.append(float(np.linalg.norm(protos[ci] - protos[cj])))
    return float(np.mean(dists)) if dists else 0.0


def compute_proto_structure_metrics(
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target: np.ndarray,
    y_target: np.ndarray,
    num_classes: int,
    kmeans_ari: float,
    kmeans_silhouette: float,
    min_samples_per_domain: int = 1,
) -> dict:
    """
    Compute structure-preserving reporting metrics (not used in training loss).
    """
    z_source = np.asarray(z_source, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)
    y_source = np.asarray(y_source, dtype=np.int64)
    y_target = np.asarray(y_target, dtype=np.int64)

    src_p = _class_prototypes(z_source, y_source, num_classes, min_samples_per_domain)
    tgt_p = _class_prototypes(z_target, y_target, num_classes, min_samples_per_domain)

    gaps = []
    for c in set(src_p.keys()) & set(tgt_p.keys()):
        gaps.append(float(np.linalg.norm(src_p[c] - tgt_p[c])))

    classwise_domain_gap_mean = float(np.mean(gaps)) if gaps else 0.0
    classwise_domain_gap_median = float(np.median(gaps)) if gaps else 0.0

    same_class_cross_domain_proto_distance = classwise_domain_gap_mean

    src_margin = _inter_class_margin(src_p)
    tgt_margin = _inter_class_margin(tgt_p)
    combined_margin = float(np.mean([src_margin, tgt_margin])) if (src_margin or tgt_margin) else 0.0

    ari_norm = float(np.clip(kmeans_ari, 0.0, 1.0))
    sil_norm = float(np.clip(kmeans_silhouette, 0.0, 1.0)) if kmeans_silhouette is not None else 0.0
    margin_norm = float(np.clip(combined_margin / (combined_margin + 1.0), 0.0, 1.0))
    structure_retention_score = float((ari_norm + sil_norm + margin_norm) / 3.0)

    return {
        "source_inter_class_proto_margin": src_margin,
        "target_inter_class_proto_margin": tgt_margin,
        "combined_inter_class_proto_margin": combined_margin,
        "same_class_cross_domain_proto_distance": same_class_cross_domain_proto_distance,
        "classwise_domain_gap_mean": classwise_domain_gap_mean,
        "classwise_domain_gap_median": classwise_domain_gap_median,
        "structure_retention_score": structure_retention_score,
    }
