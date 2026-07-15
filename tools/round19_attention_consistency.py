"""Fold/member consistency metrics for Stage 19G atom attention."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from tools.round19_attention_ensemble import validate_attention_long


def normalize_attention(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("attention vector must be finite, non-empty, and one-dimensional")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        raise ValueError("attention vector has zero mass")
    return array / total


def attention_entropy(values: Iterable[float]) -> float:
    p = normalize_attention(values)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def jensen_shannon_divergence(left: Iterable[float], right: Iterable[float]) -> float:
    p, q = normalize_attention(left), normalize_attention(right)
    if p.shape != q.shape:
        raise ValueError("JSD vectors must have equal shape")
    midpoint = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def topk_overlap(left: Iterable[float], right: Iterable[float], k: int) -> float:
    p, q = normalize_attention(left), normalize_attention(right)
    if p.shape != q.shape:
        raise ValueError("top-k vectors must have equal shape")
    count = min(max(int(k), 1), len(p))
    # Stable index tie-break makes results deterministic.
    top_p = set(np.argsort(-p, kind="stable")[:count])
    top_q = set(np.argsort(-q, kind="stable")[:count])
    return len(top_p & top_q) / count


def spearman_attention(left: Iterable[float], right: Iterable[float]) -> float:
    p, q = normalize_attention(left), normalize_attention(right)
    if p.shape != q.shape:
        raise ValueError("Spearman vectors must have equal shape")
    value = pd.Series(p).rank(method="average").corr(
        pd.Series(q).rank(method="average"), method="pearson"
    )
    return float(value) if pd.notna(value) else 1.0 if np.array_equal(p, q) else 0.0


def pairwise_member_consistency(
    attention: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    df = validate_attention_long(attention)
    rows = []
    for (candidate, eval_row), group in df.groupby(
        ["candidate_id", "eval_row_id"], sort=False
    ):
        vectors = {
            str(member): member_group.sort_values("atom_index")["attention"].to_numpy()
            for member, member_group in group.groupby("member_id", sort=True)
        }
        for left_id, right_id in combinations(sorted(vectors), 2):
            left, right = vectors[left_id], vectors[right_id]
            rows.append(
                {
                    "candidate_id": candidate,
                    "eval_row_id": eval_row,
                    "member_left": left_id,
                    "member_right": right_id,
                    "spearman": spearman_attention(left, right),
                    "topk_overlap": topk_overlap(left, right, top_k),
                    "jsd": jensen_shannon_divergence(left, right),
                    "entropy_left": attention_entropy(left),
                    "entropy_right": attention_entropy(right),
                    "top_k": min(int(top_k), len(left)),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "attention_entropy",
    "jensen_shannon_divergence",
    "normalize_attention",
    "pairwise_member_consistency",
    "spearman_attention",
    "topk_overlap",
]
