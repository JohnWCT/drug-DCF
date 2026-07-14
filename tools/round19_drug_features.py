"""Round 19 MACCS-only drug feature utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


EXPECTED_MACCS_BITS = 166


def _parse_maccs_bits(raw) -> np.ndarray:
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    else:
        text = str(raw).strip()
        if not text:
            raise ValueError("Empty MACCS bit string")
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip() != ""]
        arr = np.asarray([float(p) for p in parts], dtype=np.float32)
    # RDKit MACCSKeysFingerprint is often length 167 with unused bit0.
    if arr.size == EXPECTED_MACCS_BITS + 1:
        arr = arr[1:]
    if arr.size != EXPECTED_MACCS_BITS:
        raise ValueError(f"Expected {EXPECTED_MACCS_BITS} MACCS bits, got {arr.size}")
    return arr


def load_maccs_by_drug_name(
    drug_csv_path: str,
    *,
    drug_names: Optional[Iterable[str]] = None,
    drug_col: str = "DRUG_NAME",
    maccs_col: str = "MACCS166bits",
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(drug_csv_path)
    if drug_col not in df.columns:
        raise KeyError(f"Missing {drug_col} in {drug_csv_path}")
    if maccs_col not in df.columns:
        raise KeyError(f"Missing {maccs_col} in {drug_csv_path}")

    wanted: Optional[Set[str]] = set(str(x) for x in drug_names) if drug_names is not None else None
    out: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        name = str(row[drug_col])
        if wanted is not None and name not in wanted:
            continue
        vec = _parse_maccs_bits(row[maccs_col])
        if name in out:
            if not np.allclose(out[name], vec):
                raise ValueError(f"Duplicate DRUG_NAME with conflicting MACCS: {name}")
            continue
        out[name] = vec
    return out


def validate_maccs_coverage(
    maccs_by_drug: Dict[str, np.ndarray],
    required_drugs: Sequence[str],
) -> Dict[str, object]:
    required = [str(x) for x in required_drugs]
    missing = sorted(set(required) - set(maccs_by_drug))
    bad_dim = []
    nan_drugs = []
    for d in required:
        if d not in maccs_by_drug:
            continue
        v = maccs_by_drug[d]
        if v.shape != (EXPECTED_MACCS_BITS,):
            bad_dim.append(d)
        if not np.isfinite(v).all():
            nan_drugs.append(d)
    if missing or bad_dim or nan_drugs:
        raise ValueError(
            f"MACCS coverage failed: missing={len(missing)} bad_dim={len(bad_dim)} nan={len(nan_drugs)}; "
            f"missing_sample={missing[:5]}"
        )
    return {
        "n_required": len(required),
        "n_available": len(maccs_by_drug),
        "bit_dim": EXPECTED_MACCS_BITS,
        "ok": True,
    }


def assert_no_graph_fields_in_maccs_batch(batch: dict) -> None:
    banned = {"edge_index", "edge_attr", "graph", "node_embeddings", "batch_index"}
    hits = sorted(banned.intersection(batch.keys()))
    if hits:
        raise AssertionError(f"MACCS batch must not carry graph fields: {hits}")
