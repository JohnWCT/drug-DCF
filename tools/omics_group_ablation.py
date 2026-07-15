#!/usr/bin/env python3
"""Locked omics-block true/zero/shuffled interventions."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

VALID_ROLES = {"O1", "O2", "O3", "O4"}
CONDITIONS = ("true", "zero", "shuffled")


def ablate_omics_blocks(
    values: np.ndarray,
    blocks: Mapping[str, Sequence[int]],
    *,
    omics_role: str,
    partition_ids: Sequence[object],
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    if str(omics_role).upper() not in VALID_ROLES:
        raise ValueError(f"omics role must be one of {sorted(VALID_ROLES)}")
    array = np.asarray(values)
    if array.ndim != 2 or len(partition_ids) != array.shape[0]:
        raise ValueError("values must be samples x features and match partition_ids")
    result = {}
    partitions = np.asarray(partition_ids)
    for block_name, raw_indices in blocks.items():
        indices = np.asarray(sorted(set(map(int, raw_indices))), dtype=int)
        zero = array.copy()
        zero[:, indices] = 0
        shuffled = array.copy()
        for partition in sorted(set(partition_ids), key=str):
            rows = np.flatnonzero(partitions == partition)
            rng = np.random.RandomState(seed)  # same seed independently within each partition
            order = rng.permutation(rows)
            shuffled[np.ix_(rows, indices)] = array[np.ix_(order, indices)]
        result[str(block_name)] = {
            "true": array.copy(),
            "zero": zero,
            "shuffled": shuffled,
        }
    return result


def omics_metadata() -> dict:
    return {
        "conditions": list(CONDITIONS),
        "unit_label": "omics_feature_block",
        "forbidden_label": "genes",
        "shuffle_scope": "within_partition_fixed_seed",
    }
