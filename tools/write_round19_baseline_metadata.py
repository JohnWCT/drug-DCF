#!/usr/bin/env python3
"""Write Round 19 baseline metadata JSON (SHA + artifact hashes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    env_map = {
        "rev-parse HEAD": "ROUND19_GIT_HEAD",
        "rev-parse origin/main": "ROUND19_GIT_ORIGIN_MAIN",
    }
    key = " ".join(args)
    if key in env_map and os.environ.get(env_map[key]):
        return os.environ[env_map[key]].strip()
    cmd = ["git", "-c", "safe.directory=*"] + list(args)
    try:
        return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        if args[:2] == ["rev-parse", "HEAD"]:
            head = Path(PROJECT_ROOT) / ".git" / "HEAD"
            ref = head.read_text().strip()
            if ref.startswith("ref:"):
                return (Path(PROJECT_ROOT) / ".git" / ref.split()[1]).read_text().strip()
            return ref
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="result/optimization_runs/round19_factorial")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    meta_dir = outdir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    dirty = os.environ.get("ROUND19_GIT_STATUS")
    if dirty is None:
        try:
            dirty = _git("status", "--porcelain")
        except Exception:  # noqa: BLE001
            dirty = "unknown"

    # tracked-only dirtiness: ignore untracked logs/cleanup
    tracked_dirty = []
    for line in (dirty or "").splitlines():
        if line.startswith("??"):
            continue
        tracked_dirty.append(line)

    manifest = outdir / "manifests" / "stage19b_drug_predictor_manifest.csv"
    eligible = outdir / "data" / "round19_eligible_response.csv"
    split = outdir / "splits" / "screening_3fold_assignments.csv"

    payload = {
        "stage19a_commit": head,
        "origin_main_sha": origin,
        "head_equals_origin_main": bool(origin) and head == origin,
        "working_tree_clean_for_tracked_files": len(tracked_dirty) == 0,
        "tracked_dirty_sample": tracked_dirty[:20],
        "round18e_external_success": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest) if manifest.is_file() else None,
        "manifest_sha256": _sha256_file(manifest) if manifest.is_file() else None,
        "manifest_n_jobs": int(sum(1 for _ in manifest.open()) - 1) if manifest.is_file() else None,
        "eligible_response_sha256": _sha256_file(eligible) if eligible.is_file() else None,
        "split_assignment_sha256": _sha256_file(split) if split.is_file() else None,
        "stage19b_omics_anchors": ["O1", "O2", "O3"],
        "stage19b_expected_jobs": 117,
    }
    path = meta_dir / "round19_baseline_git.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
