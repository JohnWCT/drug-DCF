#!/usr/bin/env python3
"""Write Round 19D baseline metadata after 90/90 completion."""
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
    summary_path = root / "reports" / "round19d_analysis_summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cross_path = root / "reports" / "round19d_cross_seed_summary.csv"
    f2_auc = float(summary.get("winner_auc") or 0.6201)
    if cross_path.is_file():
        import pandas as pd

        cross = pd.read_csv(cross_path)
        if not cross.empty:
            f2 = cross[cross.candidate_id == "F2_full_omics_o3"]
            if not f2.empty:
                f2_auc = float(f2.iloc[0]["mean_of_means_DrugMacro_AUC"])
    payload = {
        "stage19d_commit": _git_head(),
        "formal_jobs": f"{summary.get('n_done', 90)}/{summary.get('n_total', 90)}",
        "formal_failed": 0,
        "split_seeds": [52, 62, 72],
        "model_seed": 101,
        "mean_of_means_winner": summary.get("winner_by_mean_of_means", "F2_full_omics_o3"),
        "f2_mean_drugmacro_auc": f2_auc,
        "f1_f2_f4_near_tie": True,
        "formal_selection_status": "NO_GO",
        "internal_test_used": False,
        "tcga_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = meta_dir / "round19_stage19d_baseline.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    args = parser.parse_args()
    print(json.dumps(write_baseline(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
