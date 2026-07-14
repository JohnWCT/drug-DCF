#!/usr/bin/env python3
"""Write Round 19B baseline metadata JSON after 117/117 completion."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _git_head() -> str:
    env = os.environ.get("ROUND19_GIT_HEAD", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "UNKNOWN"


def write_baseline(root: Path, *, expected_jobs: int = 117, completed_jobs: int = 117) -> dict:
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit": _git_head(),
        "stage19b_expected_jobs": int(expected_jobs),
        "stage19b_completed_jobs": int(completed_jobs),
        "stage19b_best_cell": "D0__P2",
        "stage19b_best_omics": "O3",
        "selection_used_internal": False,
        "selection_used_tcga": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = meta_dir / "round19_stage19b_baseline.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Round 19B baseline JSON")
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument("--expected-jobs", type=int, default=117)
    parser.add_argument("--completed-jobs", type=int, default=117)
    args = parser.parse_args()
    out = write_baseline(Path(args.root), expected_jobs=args.expected_jobs, completed_jobs=args.completed_jobs)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
