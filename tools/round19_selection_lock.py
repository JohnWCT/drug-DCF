"""Round 19 selection lock helpers (no TCGA / internal_test metrics)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from tools.round19_manifest_validator import FORBIDDEN_SELECTION_COLS, assert_selection_frame_has_no_tcga


def scan_mapping_for_forbidden(obj: Any, *, path: str = "root") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in FORBIDDEN_SELECTION_COLS:
                raise AssertionError(f"Forbidden selection key at {path}.{k}")
            scan_mapping_for_forbidden(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            scan_mapping_for_forbidden(v, path=f"{path}[{i}]")


def write_selection_lock(payload: Dict[str, Any], path: str) -> Path:
    scan_mapping_for_forbidden(payload)
    if "candidates" in payload and isinstance(payload["candidates"], list):
        # also scan candidate metric tables if attached as frames converted to records
        pass
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def validate_ranking_csv(path: str) -> None:
    df = pd.read_csv(path)
    assert_selection_frame_has_no_tcga(df)
