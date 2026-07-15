#!/usr/bin/env python3
"""Build the immutable Round 19G experiment lock with exclusive creation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_COMMITTED_PATHS = (
    "tools/round19_stage19g_case_selector.py",
    "tools/round19_stage19g_case_manifest.py",
    "tools/round19_stage19g_experiment_lock.py",
    "config/round19_stage19g_case_selection.json",
    "config/round19_stage19g_interpretability.json",
)


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


def _git(project_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot establish local committed git state: {' '.join(args)}") from exc


def committed_repository_state(
    project_root: Path, required_paths: Iterable[str]
) -> dict[str, Any]:
    head = _git(project_root, "rev-parse", "HEAD")
    if not SHA_RE.fullmatch(head) or head.upper() == "UNKNOWN":
        raise AssertionError("Round 19G experiment lock requires a known committed HEAD")
    tracked_status = _git(
        project_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_status:
        raise AssertionError(
            "Round 19G experiment lock refuses a dirty tracked working tree: "
            + tracked_status.splitlines()[0]
        )
    committed = {}
    for relative in required_paths:
        normalized = Path(relative).as_posix()
        try:
            subprocess.check_call(
                ["git", "cat-file", "-e", f"HEAD:{normalized}"],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AssertionError(
                f"Required implementation/config is not contained in local HEAD: {normalized}"
            ) from exc
        path = project_root / normalized
        if not path.is_file():
            raise FileNotFoundError(path)
        committed[normalized] = sha256_file(path)
    return {
        "local_committed_head": head,
        "branch": _git(project_root, "branch", "--show-current") or "DETACHED",
        "tracked_working_tree_clean": True,
        "required_committed_paths": committed,
        "remote_operations_required": False,
    }


def validate_repository_attestation(
    project_root: Path,
    attestation: Mapping[str, Any],
    required_paths: Iterable[str],
) -> dict[str, Any]:
    head = str(attestation.get("local_committed_head", ""))
    if not SHA_RE.fullmatch(head) or head.upper() == "UNKNOWN":
        raise AssertionError("Repository attestation has no valid local committed HEAD")
    if attestation.get("tracked_working_tree_clean") is not True:
        raise AssertionError("Repository attestation does not prove a clean tracked tree")
    committed = attestation.get("required_committed_paths")
    if not isinstance(committed, Mapping):
        raise AssertionError("Repository attestation has no committed path hashes")
    expected_paths = {Path(value).as_posix() for value in required_paths}
    if set(committed) != expected_paths:
        raise AssertionError("Repository attestation committed-path coverage mismatch")
    for relative, expected_hash in committed.items():
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(
                f"Workspace file differs from attested local HEAD content: {relative}"
            )
    result = dict(attestation)
    result["attestation_mode"] = "explicit_host_local_git_snapshot"
    result["remote_operations_required"] = False
    return result


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _verify_final_lock(path: Path, project_root: Path) -> dict[str, Any]:
    lock = _read_object(path)
    if (
        lock.get("lock_type") != "round19_final_role_lock"
        or lock.get("schema_version") != 1
        or lock.get("immutable") is not True
    ):
        raise AssertionError("Experiment lock requires the existing immutable final role lock")
    payload = json.loads(json.dumps(lock, allow_nan=False))
    expected = payload["hashes"].pop("lock_payload_sha256")
    if expected != canonical_sha256(payload):
        raise AssertionError("Final role lock payload hash mismatch")
    checkpoints = lock["hashes"].get("checkpoint_inventory")
    if not isinstance(checkpoints, list) or len(checkpoints) != 90:
        raise AssertionError("Final role lock must pin 90 checkpoint files")
    for item in checkpoints:
        checkpoint = Path(str(item["checkpoint_path"]))
        checkpoint = checkpoint if checkpoint.is_absolute() else project_root / checkpoint
        if sha256_file(checkpoint) != item["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch: {checkpoint}")
    return lock


def build_experiment_lock(
    *,
    project_root: Path,
    case_manifest_path: Path,
    config_paths: Iterable[Path],
    final_lock_path: Path,
    task_summary_path: Path,
    required_committed_paths: Iterable[str] = DEFAULT_COMMITTED_PATHS,
    repository_attestation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    repository = (
        validate_repository_attestation(
            project_root, repository_attestation, required_committed_paths
        )
        if repository_attestation is not None
        else committed_repository_state(project_root, required_committed_paths)
    )
    import pandas as pd

    from tools.round19_stage19g_case_manifest import validate_case_manifest

    cases = validate_case_manifest(pd.read_csv(case_manifest_path))
    if cases.empty:
        raise AssertionError("Case manifest must be complete and non-empty before lock creation")
    configs = {}
    for path in config_paths:
        config = _read_object(path)
        configs[str(path.relative_to(project_root))] = {
            "sha256": sha256_file(path),
            "artifact_type": config.get("artifact_type"),
            "schema_version": config.get("schema_version"),
        }
    final_lock = _verify_final_lock(final_lock_path, project_root)
    task_summary = _read_object(task_summary_path)
    case_hash = sha256_file(case_manifest_path)
    if task_summary.get("pins", {}).get("case_manifest_sha256") != case_hash:
        raise AssertionError("Task manifests are not pinned to the completed case manifest")
    pins = task_summary["pins"]
    config_hashes = {record["artifact_type"]: record["sha256"] for record in configs.values()}
    if pins.get("config_sha256") != config_hashes.get(
        "round19_stage19g_interpretability_config"
    ):
        raise AssertionError("Task manifests are not pinned to interpretability config")
    if pins.get("case_selection_config_sha256") != config_hashes.get(
        "round19_stage19g_case_selection_config"
    ):
        raise AssertionError("Task manifests are not pinned to case-selection config")
    task_manifests = {}
    for method, record in task_summary.get("manifests", {}).items():
        path = Path(str(record["path"]))
        path = path if path.is_absolute() else project_root / path
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise AssertionError(f"Task manifest hash mismatch: {path}")
        task_manifests[method] = {
            "path": str(path.relative_to(project_root)),
            "sha256": actual,
            "tasks": int(record["tasks"]),
        }
    if set(task_manifests) != {"attention", "occlusion", "omics", "routing"}:
        raise AssertionError("All four Round 19G task manifests must be complete")
    lock = {
        "lock_type": "round19_stage19g_experiment_lock",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "repository": repository,
        "case_manifest": {
            "path": str(case_manifest_path.relative_to(project_root)),
            "sha256": case_hash,
            "rows": len(cases),
            "selection_seed": 19091,
        },
        "configs": configs,
        "final_role_lock": {
            "path": str(final_lock_path.relative_to(project_root)),
            "file_sha256": sha256_file(final_lock_path),
            "payload_sha256": final_lock["hashes"]["lock_payload_sha256"],
            "roles_immutable": True,
        },
        "checkpoint_inventory": final_lock["hashes"]["checkpoint_inventory"],
        "task_manifest_summary": {
            "path": str(task_summary_path.relative_to(project_root)),
            "sha256": sha256_file(task_summary_path),
            "payload_sha256": task_summary.get("summary_payload_sha256"),
        },
        "task_manifests": task_manifests,
        "protocol": {
            "case_shard_size": 32,
            "task_unit": "candidate_x_checkpoint_x_case_shard",
            "required_members_per_candidate": 15,
            "attention_scope": "actual_locked_P2_sources_only",
            "occlusion_scope": "all_actual_locked_sources_role_deduplicated",
            "tcga_classification": "exploratory_only",
            "tcga_primary_faithfulness_excluded": True,
            "role_changes_forbidden": True,
            "prediction_ranked_representative_selection_forbidden": True,
            "contrastive_selection_classification": "explicit_post_hoc",
        },
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    return lock


def write_exclusive(lock: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create the Round 19G experiment lock")
    parser.add_argument("--project-root", default=str(root_default))
    parser.add_argument("--case-manifest")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "config/round19_stage19g_case_selection.json",
            "config/round19_stage19g_interpretability.json",
        ],
    )
    parser.add_argument(
        "--final-lock",
        default="result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json",
    )
    parser.add_argument("--task-summary")
    parser.add_argument(
        "--repository-attestation",
        help="Host-generated local Git snapshot for Docker runs without .git mounted",
    )
    parser.add_argument(
        "--capture-repository-attestation",
        help="Write a clean local-HEAD attestation and exit (run on host)",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    if args.capture_repository_attestation:
        snapshot = committed_repository_state(project_root, DEFAULT_COMMITTED_PATHS)
        destination = rooted(args.capture_repository_attestation)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({"written": str(destination), **snapshot}, indent=2))
        return
    if not args.case_manifest or not args.task_summary or not args.output:
        parser.error("--case-manifest, --task-summary, and --output are required for lock creation")

    lock = build_experiment_lock(
        project_root=project_root,
        case_manifest_path=rooted(args.case_manifest),
        config_paths=[rooted(value) for value in args.configs],
        final_lock_path=rooted(args.final_lock),
        task_summary_path=rooted(args.task_summary),
        repository_attestation=(
            _read_object(rooted(args.repository_attestation))
            if args.repository_attestation
            else None
        ),
    )
    write_exclusive(lock, rooted(args.output))
    print(
        json.dumps(
            {
                "written": args.output,
                "lock_payload_sha256": lock["lock_payload_sha256"],
                "local_committed_head": lock["repository"]["local_committed_head"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
