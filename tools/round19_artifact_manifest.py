#!/usr/bin/env python3
"""Build a read-only Round 19H retention and portable-archive plan."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.round19_reproducibility_audit import (
    attach_canonical_hash,
    audit_symlink,
    canonical_json_hash,
    sha256_file,
    write_json,
)

RETENTION = ("KEEP", "ARCHIVE", "DISPOSABLE")
LOCK_SUFFIXES = (".lock", "_lock.json", "_manifest.json", "_manifest.csv")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")


def _relative(path: Path, root: Path) -> str | None:
    try:
        absolute = path if path.is_absolute() else root / path
        return absolute.absolute().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _looks_like_path(value: str) -> bool:
    return "/" in value or Path(value).suffix.lower() in {
        ".json",
        ".csv",
        ".pt",
        ".pth",
        ".ckpt",
        ".yaml",
        ".yml",
        ".txt",
        ".log",
    }


def _extract_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _extract_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _extract_strings(item)
    elif isinstance(value, str) and _looks_like_path(value):
        yield value


def referenced_paths(path: Path, project_root: Path) -> set[str]:
    """Return existing project-relative paths named by a JSON/CSV manifest."""
    values: list[str] = []
    if path.suffix.lower() == ".json":
        try:
            values.extend(_extract_strings(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return set()
    elif path.suffix.lower() == ".csv":
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    values.extend(
                        value for value in row.values() if value and _looks_like_path(value)
                    )
        except (csv.Error, UnicodeDecodeError):
            return set()
    result: set[str] = set()
    for value in values:
        candidate = Path(value)
        absolute = candidate if candidate.is_absolute() else project_root / candidate
        relative = _relative(absolute, project_root)
        if relative is not None and (absolute.exists() or absolute.is_symlink()):
            result.add(relative)
    return result


def reachable_from_manifests(
    project_root: Path, seeds: Iterable[Path]
) -> tuple[set[str], set[str]]:
    queue = list(seeds)
    reachable: set[str] = set()
    manifests: set[str] = set()
    while queue:
        path = queue.pop()
        relative = _relative(path, project_root)
        if relative is None or relative in reachable or not (path.exists() or path.is_symlink()):
            continue
        reachable.add(relative)
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
            manifests.add(relative)
            for reference in referenced_paths(path, project_root):
                if reference not in reachable:
                    queue.append(project_root / reference)
    return reachable, manifests


def classify_artifact(
    relative_path: str, *, protected: set[str], locked_checkpoints: set[str]
) -> tuple[str, str]:
    lowered = relative_path.lower()
    name = Path(relative_path).name.lower()
    if relative_path in locked_checkpoints:
        return "KEEP", "locked_checkpoint"
    if relative_path in protected:
        return "KEEP", "lock_or_manifest_reachable"
    if name.endswith(LOCK_SUFFIXES) or "manifest" in name:
        return "KEEP", "lock_or_manifest"
    if name.endswith(CHECKPOINT_SUFFIXES):
        return "ARCHIVE", "unlocked_checkpoint"
    if any(part in lowered for part in ("/cache/", "__pycache__", ".pytest_cache")):
        return "DISPOSABLE", "regenerable_cache"
    if name.endswith((".tmp", ".temp", ".log")):
        return "DISPOSABLE", "transient_output"
    return "ARCHIVE", "preserve_by_archive_plan"


def _locked_checkpoint_paths(lock_path: Path, project_root: Path) -> set[str]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
    paths: set[str] = set()
    for item in inventory:
        value = Path(str(item["checkpoint_path"]))
        absolute = value if value.is_absolute() else project_root / value
        relative = _relative(absolute, project_root)
        if relative is None:
            raise ValueError(f"Locked checkpoint is outside project: {value}")
        paths.add(relative)
    return paths


def build_artifact_manifest(
    project_root: Path,
    final_lock_path: Path,
    inventory_roots: Iterable[Path],
    *,
    manifest_seeds: Iterable[Path] = (),
    require_complete: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    lock_path = (
        final_lock_path if final_lock_path.is_absolute() else root / final_lock_path
    )
    locked = _locked_checkpoint_paths(lock_path, root)
    if require_complete and len(locked) != 90:
        raise AssertionError(f"Expected 90 locked checkpoints, got {len(locked)}")

    candidates: set[Path] = set()
    for value in inventory_roots:
        path = value if value.is_absolute() else root / value
        if path.is_dir() and not path.is_symlink():
            candidates.update(item for item in path.rglob("*") if not item.is_dir())
        else:
            candidates.add(path)
    automatic_seeds = [
        path
        for path in candidates
        if path.is_file()
        and (
            "manifest" in path.name.lower()
            or path.name.lower().endswith(LOCK_SUFFIXES)
        )
    ]
    seeds = [lock_path, *manifest_seeds, *automatic_seeds]
    protected, discovered_manifests = reachable_from_manifests(root, seeds)
    protected.update(locked)
    protected.update(
        relative
        for relative in (_relative(path, root) for path in seeds)
        if relative is not None
    )

    candidates.update(root / path for path in protected)

    artifacts: list[dict[str, Any]] = []
    portable_mappings: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda item: str(item)):
        relative = _relative(path, root)
        if relative is None:
            continue
        retention, reason = classify_artifact(
            relative, protected=protected, locked_checkpoints=locked
        )
        entry: dict[str, Any] = {
            "path": relative,
            "retention": retention,
            "reason": reason,
            "exists": path.exists(),
            "is_symlink": path.is_symlink(),
        }
        if path.is_symlink():
            symlink = audit_symlink(path, root)
            entry["symlink"] = symlink
            if symlink["absolute_target"]:
                mapping = {
                    "archive_path": relative,
                    "source_literal_target": symlink["literal_target"],
                    "source_resolved": symlink["resolved_source"],
                    "source_content_sha256": symlink["content_sha256"],
                    "broken": symlink["broken"],
                    "outside_project": symlink["outside_project"],
                    "operation": "plan_portable_mapping_only",
                }
                portable_mappings.append(mapping)
        elif path.is_file():
            entry.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
        artifacts.append(entry)

    counts = {label: sum(row["retention"] == label for row in artifacts) for label in RETENTION}
    copy_plan = [
        {
            "source": row["path"],
            "archive_destination": row["path"],
            "operation": "plan_only_no_copy",
        }
        for row in artifacts
        if row["retention"] in {"KEEP", "ARCHIVE"}
    ]
    payload = attach_canonical_hash(
        {
            "schema": "round19_stage19h_artifact_manifest",
            "schema_version": 1,
            "stage": "19h",
            "all_done": False,
            "read_only": True,
            "cleanup_performed": False,
            "checkpoint_copy_performed": False,
            "project_relative_archive_paths": True,
            "locked_checkpoint_count": len(locked),
            "protected_reachable_count": len(protected),
            "discovered_manifests": sorted(discovered_manifests),
            "retention_counts": counts,
            "artifacts": artifacts,
            "archive_plan": copy_plan,
            "archive_plan_sha256": canonical_json_hash(copy_plan),
        }
    )
    sidecar = attach_canonical_hash(
        {
            "schema": "round19_stage19h_portable_symlink_mapping",
            "schema_version": 1,
            "source_links_modified": False,
            "mappings": portable_mappings,
        }
    )
    return payload, sidecar


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--inventory-root", action="append", required=True)
    parser.add_argument("--manifest-seed", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--portable-mapping-output")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    manifest, sidecar = build_artifact_manifest(
        root,
        resolve(args.final_lock),
        [resolve(value) for value in args.inventory_root],
        manifest_seeds=[resolve(value) for value in args.manifest_seed],
        require_complete=not args.allow_incomplete,
    )
    if not args.dry_run:
        if args.output:
            write_json(Path(args.output), manifest)
        if args.portable_mapping_output:
            write_json(Path(args.portable_mapping_output), sidecar)
    print(json.dumps({"manifest": manifest, "portable_mapping": sidecar}, indent=2))


if __name__ == "__main__":
    main()
