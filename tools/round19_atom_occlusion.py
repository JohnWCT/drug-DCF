#!/usr/bin/env python3
"""Topology-preserving atom feature-zero perturbations for Round 19G."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

MAX_PERTURBATION_BATCH = 128
RANK_GROUPS = ("top1", "top3", "top10pct", "bottom10pct")
MATCH_LEVELS = (
    ("element", "degree", "aromatic", "ring"),
    ("element", "degree", "aromatic"),
    ("element", "degree"),
    ("element",),
    (),
)


def rank_atom_sets(scores: Sequence[float]) -> dict[str, list[int]]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty vector")
    order = np.argsort(-values, kind="stable")
    k = max(1, int(math.ceil(len(values) * 0.10)))
    return {
        "top1": order[:1].tolist(),
        "top3": order[: min(3, len(order))].tolist(),
        "top10pct": order[:k].tolist(),
        "bottom10pct": order[-k:].tolist(),
    }


def feature_zero_graph(data: Any, atom_indices: Iterable[int]) -> Any:
    """Clone data and zero selected x rows; edge topology and node identity stay fixed."""
    out = data.clone()
    indices = sorted(set(int(i) for i in atom_indices))
    if any(i < 0 or i >= int(data.x.shape[0]) for i in indices):
        raise IndexError("atom index outside graph")
    original_edges = data.edge_index.clone()
    out.x = data.x.clone()
    if indices:
        out.x[torch.as_tensor(indices, dtype=torch.long, device=out.x.device)] = 0
    if out.x.shape != data.x.shape or not torch.equal(out.edge_index, original_edges):
        raise AssertionError("feature-zero perturbation changed topology/node identity")
    return out


def batched(items: Sequence[Any], size: int = MAX_PERTURBATION_BATCH):
    if not 1 <= int(size) <= MAX_PERTURBATION_BATCH:
        raise ValueError(f"perturbation batch must be 1..{MAX_PERTURBATION_BATCH}")
    for start in range(0, len(items), int(size)):
        yield items[start : start + int(size)]


def matched_random_controls(
    target: Sequence[int],
    atom_metadata: Sequence[Mapping[str, Any]],
    *,
    repeats: int = 20,
    seed: int = 19,
) -> list[dict[str, Any]]:
    """Sample controls excluding target atoms, recording progressive match fallback."""
    target_set = set(map(int, target))
    if not target_set:
        raise ValueError("target cannot be empty")
    rng = np.random.RandomState(seed)
    controls = []
    for repeat in range(int(repeats)):
        available = set(range(len(atom_metadata))) - target_set
        chosen, fallbacks = [], []
        for source in sorted(target_set):
            candidates = sorted(available)
            used: tuple[str, ...] | None = None
            for fields in MATCH_LEVELS:
                matched = [
                    idx for idx in candidates
                    if all(atom_metadata[idx].get(key) == atom_metadata[source].get(key) for key in fields)
                ]
                if matched:
                    candidates, used = matched, fields
                    break
            if used is None:
                raise ValueError("not enough non-target atoms for matched controls")
            pick = int(rng.choice(candidates))
            chosen.append(pick)
            available.remove(pick)
            fallbacks.append("none" if len(used) == 4 else "+".join(used) or "unmatched")
        controls.append({
            "repeat": repeat,
            "atom_indices": chosen,
            "fallback_levels": fallbacks,
            "target_excluded": not bool(set(chosen) & target_set),
        })
    return controls
