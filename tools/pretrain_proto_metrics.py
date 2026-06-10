"""Report-only global prototype diagnostics from saved latent dictionaries."""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _classwise_arrays(
    latent_dict: Dict[str, np.ndarray],
    label_map: Dict[str, int],
) -> Dict[int, np.ndarray]:
    grouped: Dict[int, list] = {}
    for sample_id, vec in latent_dict.items():
        label = label_map.get(str(sample_id))
        if label is None:
            continue
        grouped.setdefault(int(label), []).append(np.asarray(vec, dtype=np.float64))
    return {k: np.vstack(v) for k, v in grouped.items() if len(v) > 0}


def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True)
    xy = x @ y.T
    k_xx = np.exp(-gamma * (xx + xx.T - 2.0 * (x @ x.T)))
    k_yy = np.exp(-gamma * (yy + yy.T - 2.0 * (y @ y.T)))
    k_xy = np.exp(-gamma * (xx + yy.T - 2.0 * xy))
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def compute_global_prototype_metrics(
    source_latent_path: str,
    target_latent_path: str,
    source_label_map: Dict[str, int],
    target_label_map: Dict[str, int],
) -> dict:
    """Compute global prototype diagnostics for one pretrain experiment."""
    with open(source_latent_path, "rb") as f:
        source_latent = pickle.load(f)
    with open(target_latent_path, "rb") as f:
        target_latent = pickle.load(f)

    source_by_class = _classwise_arrays(source_latent, source_label_map)
    target_by_class = _classwise_arrays(target_latent, target_label_map)
    shared_classes = sorted(set(source_by_class) & set(target_by_class))
    if not shared_classes:
        return {
            "classwise_mmd_mean": float("nan"),
            "same_class_cross_domain_distance": float("nan"),
            "inter_class_prototype_distance": float("nan"),
            "prototype_separation_ratio": float("nan"),
            "prototype_coverage": 0,
        }

    mmds = []
    same_class_dists = []
    prototypes = {}
    for class_id in shared_classes:
        src = source_by_class[class_id]
        tgt = target_by_class[class_id]
        mmds.append(_mmd_rbf(src, tgt))
        src_proto = src.mean(axis=0)
        tgt_proto = tgt.mean(axis=0)
        same_class_dists.append(float(np.linalg.norm(src_proto - tgt_proto)))
        prototypes[class_id] = 0.5 * (src_proto + tgt_proto)

    inter_dists = []
    class_ids = list(prototypes.keys())
    for i, c1 in enumerate(class_ids):
        for c2 in class_ids[i + 1 :]:
            inter_dists.append(float(np.linalg.norm(prototypes[c1] - prototypes[c2])))

    same_mean = float(np.mean(same_class_dists)) if same_class_dists else float("nan")
    inter_mean = float(np.mean(inter_dists)) if inter_dists else float("nan")
    ratio = inter_mean / same_mean if same_mean and np.isfinite(same_mean) and same_mean > 0 else float("nan")

    return {
        "classwise_mmd_mean": float(np.mean(mmds)) if mmds else float("nan"),
        "same_class_cross_domain_distance": same_mean,
        "inter_class_prototype_distance": inter_mean,
        "prototype_separation_ratio": float(ratio),
        "prototype_coverage": len(shared_classes),
    }


def write_prototype_metrics(exp_dir: str, metrics: dict) -> Tuple[str, str]:
    """Write prototype_metrics.csv and prototype_metrics.json under exp_dir."""
    csv_path = os.path.join(exp_dir, "prototype_metrics.csv")
    json_path = os.path.join(exp_dir, "prototype_metrics.json")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return csv_path, json_path
