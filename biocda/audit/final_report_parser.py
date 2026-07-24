"""Parse Round 24 final report for machine-checkable claims (Round 25 audit)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


def parse_round24_final_report(path: str | Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    status_m = re.search(r"\*\*狀態：\*\*\s*`([^`]+)`", text)
    champ_m = re.search(r"champion\s+\*\*`([^`]+)`\*\*", text, re.I)
    lock_m = re.search(r"\[[^\]]*\]\(([^)]*round24_final_model_lock\.json)\)", text)

    referenced: List[str] = []
    for m in re.finditer(r"\((?:\.\./)?(reports/[^)]+|docs/[^)]+)\)", text):
        referenced.append(m.group(1))
    # also bare paths in code fences
    for m in re.finditer(
        r"(reports/round24[^\s`]+|reports/round24_final_model_lock\.json|docs/AACDR[^\s`]+|configs/round24/[^\s`]+)",
        text,
    ):
        referenced.append(m.group(1).rstrip("`,"))

    referenced = sorted(set(referenced))
    return {
        "path": str(path),
        "status": status_m.group(1) if status_m else None,
        "champion_id": champ_m.group(1) if champ_m else None,
        "lock_ref": lock_m.group(1) if lock_m else None,
        "referenced_artifacts": referenced,
        "mentions_no_holdout": "NoHoldout" in text or "NoHoldout" in text,
        "mentions_stest0": "stest0" in text,
        "hard_gate_aacdr_gdsc": 0.5279 in _floats(text) or "> **0.5279**" in text,
        "hard_gate_aacdr_tcga": 0.4804 in _floats(text) or "> **0.4804**" in text,
    }


def _floats(text: str) -> List[float]:
    out = []
    for m in re.finditer(r"\b0\.\d+\b", text):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out
