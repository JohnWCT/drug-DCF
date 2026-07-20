#!/usr/bin/env python3
"""Repository sync gate for Round 21."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_GITIGNORE = [
    "__pycache__/",
    "*.py[cod]",
    "outputs/",
    "logs/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ipynb_checkpoints/",
]


def _run(cmd: List[str]) -> str:
    if cmd and cmd[0] == "git":
        cmd = ["git", "-c", f"safe.directory={ROOT}", *cmd[1:]]
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _git_tracked_files() -> List[str]:
    return _run(["git", "ls-files"]).splitlines()


def _check_gitignore() -> List[str]:
    issues: List[str] = []
    gitignore = ROOT / ".gitignore"
    if not gitignore.is_file():
        return ["missing .gitignore"]
    text = gitignore.read_text(encoding="utf-8")
    for pattern in REQUIRED_GITIGNORE:
        if pattern not in text:
            issues.append(f"gitignore_missing:{pattern}")
    return issues


def _forbidden_tracked(tracked: List[str]) -> List[str]:
    forbidden: List[str] = []
    for path in tracked:
        if "__pycache__" in path or path.endswith(".pyc"):
            forbidden.append(path)
        if path.startswith("outputs/") or path.startswith("logs/"):
            forbidden.append(path)
    return forbidden


def _architecture_name_consistency(manifest: Dict[str, Any], readme: str) -> List[str]:
    issues: List[str] = []
    arch = manifest.get("architecture_name", "")
    version = manifest.get("architecture_version", "")
    if not arch or not version:
        issues.append("architecture_manifest_missing_name_or_version")
    if arch and arch not in readme:
        issues.append("readme_missing_architecture_name")
    if version and version not in readme:
        issues.append("readme_missing_architecture_version")
    return issues


def audit_repository(
    *,
    expected_branch: str,
    architecture_manifest: Path,
    architecture_doc: Path,
    strict: bool,
) -> Dict[str, Any]:
    issues: List[str] = []
    git_commit = _run(["git", "rev-parse", "HEAD"])
    git_branch = _run(["git", "branch", "--show-current"])
    porcelain = _run(["git", "status", "--porcelain"])
    git_dirty = bool(porcelain) and porcelain != "unknown"
    tracked = _git_tracked_files()

    required_files = {
        str(architecture_doc.relative_to(ROOT)): architecture_doc.is_file(),
        str(architecture_manifest.relative_to(ROOT)): architecture_manifest.is_file(),
        "README.md": (ROOT / "README.md").is_file(),
    }
    for name, ok in required_files.items():
        if not ok:
            issues.append(f"missing_required:{name}")

    forbidden = _forbidden_tracked(tracked)
    issues.extend(_check_gitignore())

    architecture_version = ""
    if architecture_manifest.is_file():
        manifest = json.loads(architecture_manifest.read_text(encoding="utf-8"))
        architecture_version = manifest.get("architecture_version", "")
        if not architecture_version:
            issues.append("architecture_version_missing")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        issues.extend(_architecture_name_consistency(manifest, readme))
    elif strict:
        issues.append("architecture_manifest_missing")

    if git_branch != expected_branch and git_branch != "unknown":
        issues.append(f"unexpected_branch:{git_branch}!={expected_branch}")

    if git_dirty and strict and git_commit != "unknown":
        issues.append("working_tree_dirty")

    if git_commit == "unknown":
        issues.append("git_unavailable")

    hard_issues = [i for i in issues if i != "git_unavailable"]
    status = "PASS" if not hard_issues else "FAIL"
    report = {
        "status": status,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_is_dirty": git_dirty,
        "required_files": required_files,
        "forbidden_files": forbidden,
        "architecture_version": architecture_version,
        "issues": issues,
        "hard_issues": hard_issues,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-branch", default="main")
    parser.add_argument(
        "--architecture-manifest",
        type=Path,
        default=ROOT / "reports/biocda_architecture_manifest.json",
    )
    parser.add_argument(
        "--architecture-doc",
        type=Path,
        default=ROOT / "docs/biocda_architecture_finalization.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/repository_state_audit.json",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = audit_repository(
        expected_branch=args.expected_branch,
        architecture_manifest=args.architecture_manifest,
        architecture_doc=args.architecture_doc,
        strict=args.strict,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REPOSITORY_AUDIT={report['status']}")
    if report["status"] == "FAIL" and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
