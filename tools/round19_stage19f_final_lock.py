#!/usr/bin/env python3
"""Create and verify the immutable Round 19F final role lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

SCHEMA_VERSION = 1
REQUIRED_ROLES = {
    "historical_anchor",
    "source_performance_champion",
    "parsimonious_context_model",
    "cancer_shift_specialist",
    "chemical_shift_specialist",
    "source_only_domain_candidate",
    "efficient_model",
    "general_recommended_model",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _git_state(project_root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "UNKNOWN"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "working_tree_dirty": status not in {"", "UNKNOWN"},
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def build_final_lock(
    *,
    proposal_path: Path,
    inventory_path: Path,
    inventory_summary_path: Path,
    policy_path: Path,
    inference_settings_path: Path,
    approved_by: str,
    approved_at_utc: str,
    review_reference: str,
    notes: str,
    project_root: Path,
) -> dict:
    proposal = _read_object(proposal_path)
    summary = _read_object(inventory_summary_path)
    inventory = pd.read_csv(inventory_path)

    if proposal.get("lock_type") != "round19_final_role_proposal":
        raise AssertionError("Input is not a Round 19 final-role proposal")
    if proposal.get("proposal_only") is not True:
        raise AssertionError("Only an approved proposal may be locked")
    if proposal.get("single_champion") is not None:
        raise AssertionError("Round 19F final policy must not create a single champion")
    if proposal.get("selection_used_internal") is not False:
        raise AssertionError("Internal outcomes participated in role selection")
    if proposal.get("selection_used_tcga") is not False:
        raise AssertionError("TCGA outcomes participated in role selection")
    if set(proposal.get("roles", {})) != REQUIRED_ROLES:
        raise AssertionError("Proposal role schema is incomplete or unexpected")

    proposal_hash = sha256_file(proposal_path)
    inventory_hash = sha256_file(inventory_path)
    if summary.get("proposal_sha256") != proposal_hash:
        raise AssertionError("Proposal hash differs from reviewed inventory summary")
    if summary.get("inventory_sha256") != inventory_hash:
        raise AssertionError("Inventory hash differs from reviewed inventory summary")
    if summary.get("n_checkpoints") != 90:
        raise AssertionError("Reviewed inventory must contain 90 role checkpoints")
    if summary.get("required_members_per_candidate") != 15:
        raise AssertionError("Reviewed inventory must require 15 members per candidate")

    required_columns = {
        "source_candidate_id",
        "member_id",
        "checkpoint_path",
        "split_seed",
        "fold_id",
    }
    missing = required_columns - set(inventory.columns)
    if missing:
        raise KeyError(f"Inventory missing columns: {sorted(missing)}")
    if inventory.duplicated(["source_candidate_id", "member_id"]).any():
        raise AssertionError("Inventory candidate/member keys are not unique")
    group_sizes = inventory.groupby("source_candidate_id").size()
    if len(group_sizes) != 6 or not (group_sizes == 15).all():
        raise AssertionError("Expected six unique source candidates with 15 members each")

    checkpoint_hashes = []
    for row in inventory.sort_values(
        ["source_candidate_id", "split_seed", "fold_id"]
    ).itertuples(index=False):
        checkpoint_path = Path(str(row.checkpoint_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_hashes.append(
            {
                "source_candidate_id": str(row.source_candidate_id),
                "member_id": str(row.member_id),
                "checkpoint_path": str(row.checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            }
        )

    approval = {
        "decision": "approved",
        "approved_by": approved_by.strip(),
        "approved_at_utc": approved_at_utc,
        "review_reference": review_reference.strip(),
        "notes": notes.strip(),
    }
    if not all(
        approval[key]
        for key in ("approved_by", "approved_at_utc", "review_reference")
    ):
        raise ValueError("Approval identity, timestamp, and review reference are required")
    try:
        parsed = datetime.fromisoformat(approved_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("approved_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("approved_at_utc must include a timezone")

    lock = {
        "lock_type": "round19_final_role_lock",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "approval_metadata": approval,
        "repository": _git_state(project_root),
        "roles": proposal["roles"],
        "single_champion": None,
        "selection_used_internal": False,
        "selection_used_tcga": False,
        "posthoc_classification": "exploratory_post_hoc",
        "role_immutability": {
            "internal_test_may_change_roles": False,
            "tcga_may_change_roles": False,
            "routing_may_override_locked_roles": False,
        },
        "hashes": {
            "proposal_sha256": proposal_hash,
            "inventory_sha256": inventory_hash,
            "inventory_summary_sha256": sha256_file(inventory_summary_path),
            "policy_sha256": sha256_file(policy_path),
            "inference_settings_sha256": sha256_file(inference_settings_path),
            "checkpoint_inventory": checkpoint_hashes,
        },
    }
    _assert_finite(lock)
    lock["hashes"]["lock_payload_sha256"] = canonical_sha256(lock)
    return lock


def verify_final_lock(lock: Mapping[str, Any], project_root: Path) -> None:
    if lock.get("lock_type") != "round19_final_role_lock":
        raise AssertionError("Not a Round 19 final role lock")
    if lock.get("immutable") is not True:
        raise AssertionError("Final role lock must be immutable")
    expected = lock.get("hashes", {}).get("lock_payload_sha256")
    payload = json.loads(json.dumps(lock, allow_nan=False))
    payload["hashes"].pop("lock_payload_sha256", None)
    if expected != canonical_sha256(payload):
        raise AssertionError("Final role lock payload hash mismatch")
    for item in lock["hashes"]["checkpoint_inventory"]:
        path = Path(item["checkpoint_path"])
        if not path.is_absolute():
            path = project_root / path
        if sha256_file(path) != item["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch: {path}")


def write_exclusive(lock: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create immutable Round 19F role lock")
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--inventory-summary", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--inference-settings", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at-utc", required=True)
    parser.add_argument("--review-reference", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    lock = build_final_lock(
        proposal_path=Path(args.proposal),
        inventory_path=Path(args.inventory),
        inventory_summary_path=Path(args.inventory_summary),
        policy_path=Path(args.policy),
        inference_settings_path=Path(args.inference_settings),
        approved_by=args.approved_by,
        approved_at_utc=args.approved_at_utc,
        review_reference=args.review_reference,
        notes=args.notes,
        project_root=project_root,
    )
    verify_final_lock(lock, project_root)
    output = Path(args.output)
    write_exclusive(lock, output)
    print(
        json.dumps(
            {
                "written": str(output),
                "lock_payload_sha256": lock["hashes"]["lock_payload_sha256"],
                "checkpoint_hashes": len(lock["hashes"]["checkpoint_inventory"]),
                "immutable": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
