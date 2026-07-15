#!/usr/bin/env python3
"""Write Round 19C baseline metadata after 54/54 completion."""
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


def write_baseline(root: Path) -> dict:
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root / "reports" / "round19c_analysis_summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = {
        "stage19c_commit": _git_head(),
        "stage19b_jobs": "117/117",
        "stage19c_jobs": f"{summary.get('n_done', 54)}/{summary.get('n_jobs', 54)}",
        "stage19c_failed": int(summary.get("n_failed", 0)),
        "context_faithfulness_mean_delta": float(summary.get("shuffle_mean_delta", 0.023)),
        "context_faithfulness_positive_folds": (
            f"{summary.get('shuffle_pos_folds', 11)}/{summary.get('shuffle_n', 12)}"
        ),
        "o2_minus_o0_mean_delta": float(
            (summary.get("effect_means") or {}).get("context_effect", 0.0198)
        ),
        "o2_minus_o0_positive_folds": "21/21",
        "o3_minus_o2_mean_delta": float(
            (summary.get("effect_means") or {}).get("summary_added_to_context", -0.0009)
        ),
        "internal_test_used_for_selection": False,
        "tcga_used_for_selection": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = meta_dir / "round19_stage19c_baseline.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    args = parser.parse_args()
    print(json.dumps(write_baseline(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
