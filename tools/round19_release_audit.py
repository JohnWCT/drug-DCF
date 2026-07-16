#!/usr/bin/env python3
"""Round 19 release audit for public-reconstruction alignment.

Wraps the Stage 19H reproducibility primitives and adds policy / selection /
manifest completeness checks.  Does not rewrite the immutable final role lock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from tools.round19_deployment_policy_export import build_deployment_policy
from tools.round19_reproducibility_audit import (
    build_reproducibility_audit,
    sha256_file,
    write_json,
)
from tools.round19_registry import registry_snapshot, validate_registry_invariants
from tools.round19_role_lock import load_role_lock, role_candidate_map
from tools.round19_schema import validate_selection_input_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
)
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/reports/round19_deployment_policy.json"
)
DEFAULT_ROOT = PROJECT_ROOT / "result/optimization_runs/round19_factorial"

EXIT_PASS = 0
EXIT_INCOMPLETE = 2
EXIT_REPRO = 3
EXIT_LEAKAGE = 4
EXIT_POLICY = 5


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def audit_policy_roles(
    policy: Mapping[str, Any], lock: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    roles = role_candidate_map(lock)
    for rule in policy.get("rules", []):
        role = str(rule.get("route_to_role"))
        if role not in roles:
            failures.append(f"policy_role_missing_in_lock:{role}")
        elif roles[role] is None:
            failures.append(f"policy_role_null_candidate:{role}")
    fallback = policy.get("fallback_role")
    if fallback is not None and str(fallback) not in roles:
        failures.append(f"fallback_role_missing:{fallback}")
    if policy.get("selection_used_internal") or policy.get("selection_used_tcga"):
        failures.append("policy_marks_forbidden_selection")
    return failures


def _assert_attestation_flags_safe(payload: Mapping[str, Any], *, trail: str = "root") -> None:
    """Boolean attestation flags may mention TCGA/internal only when False/null."""
    from tools.round19_schema import ALLOWED_SELECTION_ATTESTATION_KEYS

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{trail}.{key}"
            if key in ALLOWED_SELECTION_ATTESTATION_KEYS and key.endswith("_used"):
                if value not in (False, None, 0):
                    raise AssertionError(
                        f"Attestation flag {path} must be false/null for release audit, got {value!r}"
                    )
            if isinstance(value, (dict, list)):
                _assert_attestation_flags_safe(value, trail=path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_attestation_flags_safe(value, trail=f"{trail}[{index}]")


def audit_forbidden_selection_artifacts(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if not path.is_file():
            failures.append(f"missing_selection_artifact:{path}")
            continue
        try:
            payload = _read_json(path)
            validate_selection_input_columns(payload.keys())
            _assert_attestation_flags_safe(payload)
        except AssertionError as exc:
            failures.append(f"selection_leakage:{path.name}:{exc}")
        except Exception as exc:  # noqa: BLE001 - audit should capture parse failures
            failures.append(f"selection_artifact_unreadable:{path.name}:{type(exc).__name__}")
    return failures


def build_release_audit(
    *,
    project_root: Path,
    round_root: Path,
    role_lock_path: Path,
    policy_path: Path | None,
    repository_attestation: Path | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    lock = load_role_lock(role_lock_path)
    include = [
        round_root / "reports",
        round_root / "manifests",
    ]
    attestation = None
    if repository_attestation and repository_attestation.is_file():
        attestation = _read_json(repository_attestation)

    repro = build_reproducibility_audit(
        project_root,
        role_lock_path,
        require_complete=require_complete,
        additional_paths=include,
        repository_attestation=attestation,
    )

    if policy_path is not None and policy_path.is_file():
        policy = _read_json(policy_path)
    else:
        policy = build_deployment_policy(role_lock=lock)

    selection_candidates = [
        round_root / "reports" / "round19_final_role_proposal.json",
        round_root / "reports" / "round19_stage19c_candidate_lock.json",
        round_root / "reports" / "round19_stage19e_candidate_lock.json",
    ]
    # Some locks live under metadata/stage dirs; include if present.
    for path in sorted((round_root / "reports").glob("*candidate_lock*.json")):
        if path not in selection_candidates:
            selection_candidates.append(path)

    failures = list(repro.get("failure_reasons", []))
    policy_failures = audit_policy_roles(policy, lock)
    leakage_failures = audit_forbidden_selection_artifacts(
        [path for path in selection_candidates if path.is_file()]
    )
    registry = validate_registry_invariants()

    inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
    missing_role_checkpoints = 0 if len(inventory) == 90 else 1
    if missing_role_checkpoints:
        failures.append("locked_checkpoint_count_not_90")

    status = "pass"
    exit_code = EXIT_PASS
    if leakage_failures:
        status = "fail"
        exit_code = EXIT_LEAKAGE
        failures.extend(leakage_failures)
    elif policy_failures:
        status = "fail"
        exit_code = EXIT_POLICY
        failures.extend(policy_failures)
    elif repro.get("status") != "pass" or missing_role_checkpoints:
        status = "fail"
        exit_code = EXIT_REPRO
    elif not require_complete:
        status = "incomplete"
        exit_code = EXIT_INCOMPLETE

    payload = {
        "schema": "round19_release_audit",
        "schema_version": 1,
        "status": status,
        "exit_code": exit_code,
        "ROUND19_RELEASE_AUDIT": status.upper(),
        "selection_leakage": len(leakage_failures),
        "missing_role_checkpoints": missing_role_checkpoints,
        "hash_mismatch": 0 if repro.get("status") == "pass" else 1,
        "incomplete_required_jobs": 0,
        "role_lock_path": str(role_lock_path),
        "role_lock_sha256": sha256_file(role_lock_path),
        "policy": {
            "path": str(policy_path) if policy_path else None,
            "rules": len(policy.get("rules", [])),
            "failures": policy_failures,
        },
        "registry": registry,
        "registry_snapshot": registry_snapshot(),
        "reproducibility": {
            "status": repro.get("status"),
            "failure_reasons": repro.get("failure_reasons", []),
            "git": repro.get("git"),
        },
        "failure_reasons": sorted(set(failures)),
        "remote_sync_required": False,
        "all_done": status == "pass",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--role-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--repository-attestation")
    parser.add_argument("--output")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    audit = build_release_audit(
        project_root=Path(args.project_root),
        round_root=Path(args.root),
        role_lock_path=Path(args.role_lock),
        policy_path=policy_path if policy_path.is_file() or args.strict else None,
        repository_attestation=(
            Path(args.repository_attestation) if args.repository_attestation else None
        ),
        require_complete=not args.allow_incomplete,
    )
    if args.output:
        write_json(Path(args.output), audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(int(audit["exit_code"]))


if __name__ == "__main__":
    main()
