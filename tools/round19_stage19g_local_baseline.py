#!/usr/bin/env python3
"""Build local-only repository baseline metadata for the Round 19G gate."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_SNAPSHOT_FIELDS = {
    "local_head",
    "branch",
    "tracked_working_tree_clean",
    "untracked_present",
}


def _git(project_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot determine local git state: {' '.join(args)}") from exc


def capture_local_snapshot(project_root: Path) -> dict[str, Any]:
    """Use local git plumbing only; no remote refs or network operations."""
    tracked = _git(project_root, "status", "--porcelain", "--untracked-files=no")
    untracked = _git(
        project_root, "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    branch = _git(project_root, "branch", "--show-current") or "DETACHED"
    return {
        "local_head": _git(project_root, "rev-parse", "HEAD"),
        "branch": branch,
        "tracked_working_tree_clean": tracked == "",
        "untracked_present": bool(untracked),
        "tracked_status_entries": tracked.splitlines(),
        "untracked_sample": untracked[:20],
    }


def validate_host_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_SNAPSHOT_FIELDS - set(value)
    if missing:
        raise KeyError(f"Host snapshot missing fields: {sorted(missing)}")
    head = value["local_head"]
    branch = value["branch"]
    if not isinstance(head, str) or not SHA_RE.fullmatch(head):
        raise ValueError("Host snapshot local_head must be a full 40-character SHA")
    if not isinstance(branch, str) or not branch.strip() or branch.upper() == "UNKNOWN":
        raise ValueError("Host snapshot branch must be known")
    for key in ("tracked_working_tree_clean", "untracked_present"):
        if type(value[key]) is not bool:
            raise TypeError(f"Host snapshot {key} must be an explicit boolean")
    if any(
        str(value.get(key, "")).strip().upper() == "UNKNOWN"
        for key in REQUIRED_SNAPSHOT_FIELDS
    ):
        raise ValueError("UNKNOWN cannot be interpreted as a clean host snapshot")
    result = dict(value)
    result.setdefault("tracked_status_entries", [])
    result.setdefault("untracked_sample", [])
    return result


def build_local_baseline(
    project_root: Path, *, host_snapshot: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    snapshot = (
        validate_host_snapshot(host_snapshot)
        if host_snapshot is not None
        else validate_host_snapshot(capture_local_snapshot(project_root))
    )
    return {
        "artifact_type": "round19_stage19g_local_baseline",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_source": "explicit_host_snapshot" if host_snapshot is not None else "local_repo",
        **snapshot,
        "remote_sync_required": False,
        "remote_operations_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local-only Round 19G baseline")
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--host-snapshot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    host_snapshot = None
    if args.host_snapshot:
        value = json.loads(Path(args.host_snapshot).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Host snapshot must be a JSON object")
        host_snapshot = value
    payload = build_local_baseline(
        Path(args.project_root), host_snapshot=host_snapshot
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
