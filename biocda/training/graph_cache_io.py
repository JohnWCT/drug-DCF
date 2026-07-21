"""Persist drug graph cache for parallel XA training workers."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from biocda.data.xa_dataset import build_xa_dataset


def graph_cache_path(root: Path) -> Path:
    return root / "shared_graph_cache.pkl"


def ensure_graph_cache(
    *,
    dev_rows_path: Path,
    feature_dir: str,
    drug_smiles_path: str,
    cache_root: Path,
    force_rebuild: bool = False,
) -> Path:
    path = graph_cache_path(cache_root)
    if path.is_file() and not force_rebuild:
        return path

    cache_root.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(dev_rows_path)
    graph_cache: Dict[str, Any] = {}
    build_xa_dataset(
        dev,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
        graph_cache=graph_cache,
    )
    with path.open("wb") as f:
        pickle.dump(graph_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_graph_cache(cache_root: Path) -> Dict[str, Any]:
    path = graph_cache_path(cache_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing graph cache: {path}")
    with path.open("rb") as f:
        return pickle.load(f)
