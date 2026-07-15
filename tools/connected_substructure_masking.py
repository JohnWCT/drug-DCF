#!/usr/bin/env python3
"""Connected-subgraph masks and matched connected controls for Round 19G."""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np

FRACTIONS = (0.03, 0.05, 0.10)


def adjacency(edge_index, num_nodes: int) -> list[set[int]]:
    graph = [set() for _ in range(int(num_nodes))]
    edges = edge_index.detach().cpu().numpy() if hasattr(edge_index, "detach") else np.asarray(edge_index)
    if edges.size:
        if edges.shape[0] != 2:
            edges = edges.T
        for left, right in edges.T:
            graph[int(left)].add(int(right))
            graph[int(right)].add(int(left))
    return graph


def connected_mask(
    scores: Sequence[float], edge_index, fraction: float, *, allowed: set[int] | None = None
) -> list[int]:
    if float(fraction) not in FRACTIONS:
        raise ValueError(f"fraction must be one of {FRACTIONS}")
    n = len(scores)
    candidates = set(range(n)) if allowed is None else set(allowed)
    if not candidates:
        raise ValueError("no eligible nodes")
    size = min(len(candidates), max(1, int(math.ceil(n * fraction))))
    graph = adjacency(edge_index, n)
    seed = max(candidates, key=lambda idx: (float(scores[idx]), -idx))
    selected, queued = [], {seed}
    frontier = deque([seed])
    while frontier and len(selected) < size:
        node = frontier.popleft()
        selected.append(node)
        neighbors = sorted(
            (graph[node] & candidates) - set(selected) - queued,
            key=lambda idx: (-float(scores[idx]), idx),
        )
        frontier.extend(neighbors)
        queued.update(neighbors)
    if len(selected) != size:
        raise ValueError("eligible component is too small")
    return selected


def matched_connected_random(
    target: Sequence[int], edge_index, num_nodes: int, *, repeats: int = 20, seed: int = 19
) -> list[dict]:
    target_set = set(map(int, target))
    allowed = set(range(num_nodes)) - target_set
    rng = np.random.RandomState(seed)
    rows = []
    for repeat in range(int(repeats)):
        random_scores = rng.uniform(size=num_nodes)
        mask = connected_mask(
            random_scores, edge_index, min(FRACTIONS, key=lambda x: abs(math.ceil(num_nodes*x)-len(target))),
            allowed=allowed,
        )
        if len(mask) != len(target):
            # Grow deterministically to exact target size when percentage rounding differs.
            graph = adjacency(edge_index, num_nodes)
            while len(mask) < len(target):
                options = sorted(set().union(*(graph[i] for i in mask)) & allowed - set(mask))
                if not options:
                    raise ValueError("cannot construct matched connected random control")
                mask.append(int(rng.choice(options)))
            mask = mask[:len(target)]
        rows.append({"repeat": repeat, "atom_indices": mask, "target_excluded": True})
    return rows
