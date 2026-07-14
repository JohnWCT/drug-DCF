"""Round 19C context shuffle controls (ModelID-level derangement)."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

BASE_SEED = 19031


def shuffle_seeds_for_fold(fold_id: int) -> Tuple[int, int]:
    """Return (train_shuffle_seed, validation_shuffle_seed) for a fold."""
    fid = int(fold_id)
    train_seed = BASE_SEED + fid * 100 + 1
    val_seed = BASE_SEED + fid * 100 + 2
    return train_seed, val_seed


def build_modelid_derangement(
    model_ids: Sequence[str],
    seed: int,
    *,
    require_derangement: bool = True,
) -> Dict[str, str]:
    """Deterministic ModelID -> donor ModelID map with no self-mapping."""
    ids = sorted({str(m) for m in model_ids})
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        if require_derangement:
            raise ValueError("Cannot build derangement for a single ModelID partition")
        return {ids[0]: ids[0]}

    rng = random.Random(int(seed))
    for _ in range(10000):
        perm = ids[:]
        rng.shuffle(perm)
        if require_derangement and any(a == b for a, b in zip(ids, perm)):
            continue
        return dict(zip(ids, perm))
    raise RuntimeError(f"Failed to build derangement for n={n} seed={seed}")


def build_partition_permutation(model_ids: Sequence[str], seed: int) -> Dict[str, str]:
    """Alias for within-partition ModelID derangement."""
    return build_modelid_derangement(model_ids, seed, require_derangement=True)


def context_slice_for_omics(omics_id: str) -> Tuple[int, int]:
    """Return (z_end, ctx_end) indices for context replacement."""
    oid = str(omics_id).upper()
    if oid in {"O2", "O2_SHUFFLED"}:
        return 64, 80
    if oid in {"O3", "O3_SHUFFLED"}:
        return 64, 80
    raise ValueError(f"Context shuffle unsupported for omics_id={omics_id}")


def apply_context_permutation(
    vec: np.ndarray,
    donor_vec: np.ndarray,
    omics_id: str,
) -> np.ndarray:
    """Keep Z (and O3 summary); replace context slice from donor."""
    out = np.asarray(vec, dtype=np.float32).reshape(-1).copy()
    donor = np.asarray(donor_vec, dtype=np.float32).reshape(-1)
    z_end, ctx_end = context_slice_for_omics(omics_id)
    if out.shape[0] < ctx_end or donor.shape[0] < ctx_end:
        raise ValueError(
            f"Vector too short for omics={omics_id}: got {out.shape[0]}, need {ctx_end}"
        )
    out[z_end:ctx_end] = donor[z_end:ctx_end]
    return out


def validate_context_shuffle(
    perm: Dict[str, str],
    allowed_ids: Iterable[str],
    *,
    require_derangement: bool = True,
) -> None:
    """Assert permutation is within-partition, complete, and deranged."""
    allowed = {str(x) for x in allowed_ids}
    keys = {str(k) for k in perm.keys()}
    if keys != allowed:
        missing = sorted(allowed - keys)
        extra = sorted(keys - allowed)
        raise AssertionError(
            f"Permutation keys mismatch allowed_ids: missing={missing[:5]} extra={extra[:5]}"
        )
    for src, donor in perm.items():
        ds = str(donor)
        if ds not in allowed:
            raise AssertionError(f"Donor {ds} not in allowed partition for src={src}")
        if require_derangement and str(src) == ds:
            raise AssertionError(f"Self-mapping forbidden in derangement: {src}")


__all__ = [
    "BASE_SEED",
    "apply_context_permutation",
    "build_modelid_derangement",
    "build_partition_permutation",
    "context_slice_for_omics",
    "shuffle_seeds_for_fold",
    "validate_context_shuffle",
]
