"""Deterministic true/shuffled/zero context controls for Stage 19G."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from tools.round19_context_controls import (
    BASE_SEED,
    apply_context_permutation,
    build_partition_permutation,
    context_slice_for_omics,
    validate_context_shuffle,
)


CONTEXT_CONDITIONS = ("true", "shuffled", "zero")


def build_partition_context_controls(
    partitions: Mapping[str, Sequence[str]],
    *,
    seed: int = BASE_SEED + 1907,
) -> Dict[str, Dict[str, str]]:
    """Build independent deterministic derangements; donors never cross partitions."""
    result: Dict[str, Dict[str, str]] = {}
    seen: set[str] = set()
    for offset, partition in enumerate(sorted(map(str, partitions))):
        ids = [str(value) for value in partitions[partition]]
        overlap = seen & set(ids)
        if overlap:
            raise AssertionError(f"ModelIDs occur in multiple partitions: {sorted(overlap)[:5]}")
        seen.update(ids)
        permutation = build_partition_permutation(ids, int(seed) + offset)
        validate_context_shuffle(permutation, ids)
        result[partition] = permutation
    return result


def context_condition_vector(
    vector: np.ndarray,
    *,
    condition: str,
    omics_id: str,
    model_id: str,
    latent_by_id: Mapping[str, np.ndarray],
    partition_permutation: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Return one control vector without changing latent source storage."""
    mode = str(condition).lower()
    if mode not in CONTEXT_CONDITIONS:
        raise ValueError(f"Unknown context condition={condition!r}")
    base = np.asarray(vector, dtype=np.float32).reshape(-1).copy()
    if mode == "true":
        return base
    if mode == "zero":
        start, end = context_slice_for_omics(omics_id)
        base[start:end] = 0.0
        return base
    if partition_permutation is None:
        raise ValueError("shuffled context requires an explicit within-partition permutation")
    donor_id = str(partition_permutation.get(str(model_id), ""))
    if not donor_id or donor_id not in latent_by_id:
        raise KeyError(f"Missing shuffled-context donor for ModelID={model_id}")
    return apply_context_permutation(base, latent_by_id[donor_id], omics_id)


__all__ = [
    "CONTEXT_CONDITIONS",
    "build_partition_context_controls",
    "context_condition_vector",
]
