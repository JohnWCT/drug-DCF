"""Round 18 sklearn compatibility helpers."""
from __future__ import annotations

import warnings

import sklearn


def _parse_version(v: str):
    parts = []
    for p in str(v).split(".")[:3]:
        try:
            parts.append(int("".join(ch for ch in p if ch.isdigit()) or "0"))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def require_stratified_group_kfold(min_version: str = "1.3.2"):
    """
    Import StratifiedGroupKFold with a Round-18 version warning.

    Round 18 officially pins scikit-learn==1.3.2 via requirements-round18.txt.
    Do not silently alter the base Dockerfile used by Round 1–17.
    """
    current = sklearn.__version__
    if _parse_version(current) < _parse_version(min_version):
        warnings.warn(
            f"Round 18 recommends scikit-learn>={min_version} "
            f"(found {current}). Install with: pip install -r requirements-round18.txt",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "StratifiedGroupKFold is unavailable. "
            "Install Round 18 deps: pip install -r requirements-round18.txt"
        ) from exc
    return StratifiedGroupKFold


def sklearn_version() -> str:
    return sklearn.__version__
