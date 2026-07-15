#!/usr/bin/env python3
"""Read-only compatibility adapter from the Round 19F lock to 19G routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.round19_deployment_policy import select_role
from tools.round19_stage19f_final_lock import REQUIRED_ROLES, sha256_file, verify_final_lock


def _resolve_source_id(requested: object, available: set[str]) -> str:
    value = str(requested or "").strip()
    if not value:
        raise AssertionError("Locked role has no source_candidate_id")
    matches = sorted(
        source
        for source in available
        if source == value
        or source.startswith(value + "_")
        or value.startswith(source + "_")
    )
    if len(matches) != 1:
        raise AssertionError(
            f"Locked source candidate {value!r} does not resolve uniquely: {matches}"
        )
    return matches[0]


def load_verified_lock(lock_path: Path, project_root: Path) -> dict[str, Any]:
    """Verify schema-v1 payload and every checkpoint without mutating the lock."""
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise TypeError("Final lock must be a JSON object")
    if lock.get("lock_type") == "round19_final_role_proposal":
        raise AssertionError("Proposal roles are not accepted by the 19G lock adapter")
    if (
        lock.get("lock_type") != "round19_final_role_lock"
        or lock.get("schema_version") != 1
        or lock.get("immutable") is not True
    ):
        raise AssertionError("19G requires the immutable schema-v1 final role lock")
    if "status" in lock and lock["status"] == "LOCKED":
        raise AssertionError("Schema-v1 lock authority is immutable=true, not status=LOCKED")
    if set(lock.get("roles", {})) != REQUIRED_ROLES:
        raise AssertionError("Final lock role schema is incomplete or unexpected")
    inventory = lock.get("hashes", {}).get("checkpoint_inventory")
    if not isinstance(inventory, list) or len(inventory) != 90:
        raise AssertionError("Final lock must pin exactly 90 checkpoints")
    verify_final_lock(lock, Path(project_root))
    lock["_lock_file_sha256"] = sha256_file(lock_path)
    return lock


def route_locked(
    lock_path: Path,
    novelty: object,
    *,
    project_root: Path,
    proposal_roles: object = None,
) -> dict[str, Any]:
    """Route through the existing deterministic selector, then resolve the lock."""
    if proposal_roles is not None:
        raise TypeError("proposal_roles is forbidden; pass the final lock path only")
    lock = load_verified_lock(Path(lock_path), Path(project_root))
    role = select_role(novelty)
    role_record = lock["roles"].get(role)
    if not isinstance(role_record, Mapping):
        raise AssertionError(f"Selected role is absent from final lock: {role}")
    available = {
        str(item["source_candidate_id"])
        for item in lock["hashes"]["checkpoint_inventory"]
    }
    source = _resolve_source_id(role_record.get("source_candidate_id"), available)
    return {
        "selected_role": role,
        "candidate_id": role_record.get("candidate_id"),
        "source_candidate_id": source,
        "lock_file_sha256": lock["_lock_file_sha256"],
        "lock_payload_sha256": lock["hashes"]["lock_payload_sha256"],
        "lock_schema_version": lock["schema_version"],
        "lock_immutable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Route Round 19G through the final role lock")
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--novelty-class", required=True)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args()
    print(
        json.dumps(
            route_locked(
                Path(args.final_lock),
                args.novelty_class,
                project_root=Path(args.project_root),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
