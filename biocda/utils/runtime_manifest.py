"""Run manifest helpers for xa_validation outputs."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def git_commit() -> str:
    try:
        proc = _git_cmd("rev-parse", "HEAD")
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        proc = _git_cmd("status", "--porcelain")
        return bool(proc.stdout.strip()) if proc.returncode == 0 else False
    except Exception:  # noqa: BLE001
        return False


def build_run_manifest(*, command: str, config: Dict[str, Any], config_hash: str, seed: int) -> Dict[str, Any]:
    return {
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash,
        "random_seed": seed,
        "config": config,
    }


def write_run_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
