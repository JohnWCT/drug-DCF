"""Persist drug graph cache for parallel XA training workers."""
from __future__ import annotations

import os
import pickle
import tempfile
import time
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
    if path.is_file() and path.stat().st_size > 0 and not force_rebuild:
        try:
            with path.open("rb") as f:
                pickle.load(f)
            return path
        except Exception:
            force_rebuild = True

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "shared_graph_cache.lock"
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if path.is_file() and path.stat().st_size > 0:
                try:
                    with path.open("rb") as f:
                        pickle.load(f)
                    return path
                except Exception:
                    pass
            time.sleep(0.5)

    try:
        if path.is_file() and path.stat().st_size > 0 and not force_rebuild:
            try:
                with path.open("rb") as f:
                    pickle.load(f)
                return path
            except Exception:
                pass
        dev = pd.read_csv(dev_rows_path)
        graph_cache: Dict[str, Any] = {}
        build_xa_dataset(
            dev,
            feature_dir=feature_dir,
            drug_smiles_path=drug_smiles_path,
            graph_cache=graph_cache,
        )
        with tempfile.NamedTemporaryFile(dir=str(cache_root), delete=False, suffix=".pkl.tmp") as tmp:
            pickle.dump(graph_cache, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_name = tmp.name
        os.replace(tmp_name, path)
        return path
    finally:
        if lock_path.exists():
            lock_path.unlink()


def load_graph_cache(cache_root: Path) -> Dict[str, Any]:
    path = graph_cache_path(cache_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing graph cache: {path}")
    with path.open("rb") as f:
        return pickle.load(f)
