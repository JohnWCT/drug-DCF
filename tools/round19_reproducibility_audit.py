#!/usr/bin/env python3
"""Deterministic, read-only Round 19H reproducibility primitives and audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHUNK_SIZE = 1 << 20
SCHEMA_VERSION = 1
ATTESTATION_KEYS = frozenset({"attestation", "created_at_utc", "observed_at_utc"})
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_payload(value: Any) -> Any:
    """Remove non-canonical attestations recursively before stable hashing."""
    if isinstance(value, Mapping):
        return {
            str(key): canonical_payload(item)
            for key, item in value.items()
            if str(key) not in ATTESTATION_KEYS
            and str(key) != "canonical_sha256"
            and not str(key).endswith("_canonical_sha256")
        }
    if isinstance(value, (list, tuple)):
        return [canonical_payload(item) for item in value]
    return value


def attach_canonical_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_json_hash(canonical_payload(result))
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def csv_fingerprint(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration:
            columns = []
        row_count = sum(1 for _ in reader)
    schema = {"columns": columns, "column_count": len(columns)}
    return {
        "raw_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "schema": schema,
        "schema_sha256": canonical_json_hash(schema),
    }


def _project_relative(path: Path, project_root: Path) -> str | None:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return None


def audit_symlink(path: Path, project_root: Path) -> dict[str, Any]:
    literal = os.readlink(path)
    target = Path(literal)
    source = target if target.is_absolute() else path.parent / target
    resolved = source.resolve(strict=False)
    broken = not source.exists()
    outside = _project_relative(resolved, project_root.resolve()) is None
    content_hash = sha256_file(resolved) if resolved.is_file() else None
    return {
        "path": _project_relative(path, project_root) or str(path),
        "literal_target": literal,
        "resolved_source": str(resolved),
        "content_sha256": content_hash,
        "broken": broken,
        "absolute_target": target.is_absolute(),
        "outside_project": outside,
    }


def tree_manifest(
    project_root: Path, paths: Iterable[Path], *, include_directories: bool = False
) -> list[dict[str, Any]]:
    """Describe paths without following directory symlinks."""
    root = project_root.resolve()
    entries: list[dict[str, Any]] = []
    expanded: set[Path] = set()
    for original in paths:
        path = original if original.is_absolute() else project_root / original
        if path.is_dir() and not path.is_symlink():
            if include_directories:
                expanded.add(path)
            expanded.update(item for item in path.rglob("*"))
        else:
            expanded.add(path)
    for path in sorted(expanded, key=lambda item: str(item)):
        relative = _project_relative(path, project_root)
        if relative is None:
            raise ValueError(f"Tree manifest path is outside project: {path}")
        if path.is_symlink():
            entry = {"type": "symlink", **audit_symlink(path, root)}
        elif path.is_file():
            entry = {
                "path": relative,
                "type": "file",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.suffix.lower() == ".csv":
                entry["csv"] = csv_fingerprint(path)
        elif path.is_dir():
            entry = {"path": relative, "type": "directory"}
        else:
            entry = {"path": relative, "type": "missing"}
        entries.append(entry)
    return entries


def _run_diagnostic(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, text=True, capture_output=True, timeout=15, check=False
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}


def collect_git_state(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = _run_diagnostic(("git", *args), project_root)
        return result.get("stdout", "") if result.get("available") else "UNKNOWN"

    commit = git("rev-parse", "HEAD") or "UNKNOWN"
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "UNKNOWN"
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status == "UNKNOWN":
        status_hash = "UNKNOWN"
        dirty: bool | None = None
    else:
        status_hash = hashlib.sha256(status.encode("utf-8")).hexdigest()
        dirty = bool(status)
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": dirty,
        "status_sha256": status_hash,
        "remote_sync_required": False,
        "valid": commit != "UNKNOWN",
        "failure_reasons": [] if commit != "UNKNOWN" else ["git_commit_unknown"],
    }


def git_state_from_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate host Git state when Docker intentionally has no ``.git`` mount."""
    commit = str(value.get("local_committed_head", ""))
    branch = str(value.get("branch", "")).strip()
    clean = value.get("tracked_working_tree_clean")
    if not SHA_RE.fullmatch(commit):
        raise ValueError("Repository attestation requires a full local commit SHA")
    if not branch or branch.upper() == "UNKNOWN":
        raise ValueError("Repository attestation requires a known branch")
    if clean is not True:
        raise AssertionError("Repository attestation requires a clean tracked tree")
    if value.get("remote_operations_required") is not False:
        raise AssertionError("Repository attestation must prohibit remote operations")
    required = value.get("required_committed_paths")
    if not isinstance(required, Mapping) or not required:
        raise ValueError("Repository attestation requires committed path hashes")
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": False,
        "status_sha256": None,
        "remote_sync_required": False,
        "valid": True,
        "source": "verified_host_repository_attestation",
        "required_committed_paths": dict(required),
        "failure_reasons": [],
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _first_package_version(*distributions: str) -> str | None:
    for distribution in distributions:
        version = _package_version(distribution)
        if version is not None:
            return version
    return None


def collect_environment_metadata() -> dict[str, Any]:
    packages = {
        "torch": _first_package_version("torch"),
        "sklearn": _first_package_version("scikit-learn"),
        "rdkit": _first_package_version("rdkit", "rdkit-pypi"),
        "pyg": _first_package_version("torch-geometric"),
    }
    torch_runtime: dict[str, Any] = {}
    try:
        import torch

        torch_runtime = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
    except (ImportError, RuntimeError) as exc:
        torch_runtime = {"available": False, "error": type(exc).__name__}
    cgroup = ""
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": torch_runtime,
        "docker": {
            "inside_container": Path("/.dockerenv").exists() or "docker" in cgroup,
            "container_hostname": platform.node(),
            "image_reference": os.environ.get("ROUND19_DOCKER_IMAGE"),
            "container_name": os.environ.get("HOSTNAME"),
            "cgroup_sha256": hashlib.sha256(cgroup.encode("utf-8")).hexdigest(),
        },
        "nvidia_smi": _run_diagnostic(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            )
        ),
    }


def _load_lock(lock_path: Path, project_root: Path, require_complete: bool) -> dict[str, Any]:
    if not lock_path.is_file():
        if require_complete:
            raise FileNotFoundError(lock_path)
        return {"path": str(lock_path), "present": False, "checkpoint_count": 0}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
    if require_complete and len(inventory) != 90:
        raise AssertionError(f"Final role lock must contain 90 checkpoints, got {len(inventory)}")
    checkpoint_entries = []
    for item in inventory:
        checkpoint = Path(str(item["checkpoint_path"]))
        absolute = checkpoint if checkpoint.is_absolute() else project_root / checkpoint
        present = absolute.is_file()
        actual_hash = sha256_file(absolute) if present else None
        matches = present and actual_hash == item.get("checkpoint_sha256")
        checkpoint_entries.append(
            {
                "path": str(item["checkpoint_path"]),
                "present": present,
                "sha256": actual_hash,
                "matches_lock": matches,
            }
        )
    if require_complete and not all(row["matches_lock"] for row in checkpoint_entries):
        raise AssertionError("One or more locked checkpoints are missing or hash-mismatched")
    return {
        "path": _project_relative(lock_path, project_root) or str(lock_path),
        "present": True,
        "sha256": sha256_file(lock_path),
        "checkpoint_count": len(inventory),
        "checkpoints": checkpoint_entries,
    }


def build_reproducibility_audit(
    project_root: Path,
    final_lock_path: Path,
    *,
    require_complete: bool = True,
    additional_paths: Iterable[Path] = (),
    repository_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    lock_path = (
        final_lock_path if final_lock_path.is_absolute() else root / final_lock_path
    )
    lock = _load_lock(lock_path, root, require_complete)
    git = (
        git_state_from_attestation(repository_attestation)
        if repository_attestation is not None
        else collect_git_state(root)
    )
    failures = list(git["failure_reasons"])
    if require_complete and lock.get("checkpoint_count") != 90:
        failures.append("locked_checkpoint_count_not_90")
    paths = [lock_path, *additional_paths]
    manifest = tree_manifest(root, paths)
    payload = {
        "schema": "round19_stage19h_reproducibility_audit",
        "schema_version": SCHEMA_VERSION,
        "stage": "19h",
        "status": "pass" if not failures else "fail",
        "all_done": False,
        "remote_sync_required": False,
        "read_only": True,
        "git": git,
        "environment": collect_environment_metadata(),
        "final_role_lock": lock,
        "tree_manifest": manifest,
        "tree_manifest_sha256": canonical_json_hash(manifest),
        "failure_reasons": sorted(failures),
    }
    return attach_canonical_hash(payload)


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root_default))
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--repository-attestation")
    parser.add_argument("--output")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    lock = Path(args.final_lock)
    if not lock.is_absolute():
        lock = root / lock
    repository_attestation = None
    if args.repository_attestation:
        repository_attestation = json.loads(
            Path(args.repository_attestation).read_text(encoding="utf-8")
        )
        if not isinstance(repository_attestation, dict):
            raise TypeError("Repository attestation must be a JSON object")
    audit = build_reproducibility_audit(
        root,
        lock,
        require_complete=not args.allow_incomplete,
        additional_paths=[Path(value) for value in args.include],
        repository_attestation=repository_attestation,
    )
    if args.output and not args.dry_run:
        write_json(Path(args.output), audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
