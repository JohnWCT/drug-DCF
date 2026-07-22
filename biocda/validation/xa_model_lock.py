"""Write BioCDA-XA lock / reject manifest for Round 23."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from biocda.utils.hashing import sha256_file
from biocda.validation.xa_selection_gate import XASelectionDecision


def build_xa_lock_manifest(
    decision: XASelectionDecision,
    *,
    architecture_version: str = "biocda-xa-v2",
    checkpoint_paths: Optional[List[str]] = None,
    git_commit: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = checkpoint_paths or []
    payload: Dict[str, Any] = {
        "status": decision.status,
        "model_name": decision.selected_model,
        "selected_training": decision.selected_training,
        "architecture_version": architecture_version,
        "architecture_candidate_name": "BioCDA-XA-Candidate",
        "failures": decision.failures,
        "gates": [{"name": g.name, "passed": g.passed, "details": g.details} for g in decision.gates],
        "checkpoint_paths": paths,
        "checkpoint_hashes": [sha256_file(Path(p)) if Path(p).is_file() else "missing" for p in paths],
        "git_commit": git_commit,
        "tcga_used_for_selection": False,
        "predictive_reference": {
            "canonical_name": "BioCDA-Predictive",
            "status": "LOCKED_REFERENCE",
            "note": "XA attention must not explain Predictive predictions if XA is REJECTED",
        },
    }
    if decision.status == "REJECTED":
        payload["reason"] = "performance_failure" if "performance_failure" in decision.failures else ";".join(decision.failures)
    if extra:
        payload.update(extra)
    return payload


def write_xa_lock_manifest(path: Path, payload: Dict[str, Any]) -> None:
    if payload.get("status") == "LOCKED" and not payload.get("model_name"):
        raise ValueError("Cannot write LOCKED manifest without model_name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
