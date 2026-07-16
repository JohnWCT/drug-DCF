#!/usr/bin/env python3
"""Public-facing Round 19 role-lock adapter over the immutable Stage 19F lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.round19_stage19f_final_lock import (
    SCHEMA_VERSION,
    verify_final_lock,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
)
REQUIRED_ROLES = (
    "historical_anchor",
    "source_performance_champion",
    "parsimonious_context_model",
    "cancer_shift_specialist",
    "chemical_shift_specialist",
    "source_only_domain_candidate",
    "efficient_model",
    "general_recommended_model",
)


def default_role_lock_path() -> Path:
    return DEFAULT_LOCK


def load_role_lock(path: Path | None = None) -> dict[str, Any]:
    lock_path = Path(path) if path else DEFAULT_LOCK
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Role lock must be a JSON object: {lock_path}")
    verify_final_lock(payload, project_root=PROJECT_ROOT)
    missing = [role for role in REQUIRED_ROLES if role not in payload.get("roles", {})]
    if missing:
        raise KeyError(f"Role lock missing roles: {missing}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError(
            f"Unexpected role-lock schema_version={payload.get('schema_version')}"
        )
    if payload.get("immutable") is not True:
        raise AssertionError("Role lock must be immutable")
    if payload.get("selection_used_internal") or payload.get("selection_used_tcga"):
        raise AssertionError("Role lock records forbidden selection leakage")
    payload["_path"] = str(lock_path)
    return payload


def role_candidate_map(lock: Mapping[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for role, record in lock["roles"].items():
        if record is None:
            out[role] = None
            continue
        if not isinstance(record, Mapping):
            raise TypeError(f"Role {role} must be an object or null")
        candidate = record.get("source_candidate_id") or record.get("candidate_id")
        out[role] = None if candidate is None else str(candidate)
    return out


def summarize_role_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
    return {
        "schema": "round19_role_lock_summary",
        "schema_version": 1,
        "lock_type": lock.get("lock_type"),
        "immutable": lock.get("immutable"),
        "single_champion": lock.get("single_champion"),
        "selection_used_internal": lock.get("selection_used_internal"),
        "selection_used_tcga": lock.get("selection_used_tcga"),
        "roles": role_candidate_map(lock),
        "checkpoint_inventory_count": len(inventory),
        "path": lock.get("_path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--summary-output")
    args = parser.parse_args()
    lock = load_role_lock(Path(args.lock))
    summary = summarize_role_lock(lock)
    if args.summary_output:
        out = Path(args.summary_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
