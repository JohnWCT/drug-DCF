"""Artifact retention policy for optimization runs."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from glob import glob
from typing import Iterable, List, Set

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LARGE_PATTERNS = ["*.pt", "*.pth", "*checkpoint*", "*latent*.pkl", "*latent*.csv"]
PROTECTED_BASENAMES = {
    "params.json",
    "resolved_config.json",
    "run_status.json",
    "gan_metrics.json",
    "gan_metrics.csv",
    "prototype_metrics.json",
    "prototype_metrics.csv",
    "pretrain_sweep_manifest.csv",
    "finetune_dispatch_manifest.csv",
}


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _matches_large_pattern(path: str) -> bool:
    base = os.path.basename(path)
    if base in PROTECTED_BASENAMES:
        return False
    if base.endswith(".csv") and "summary" in base.lower():
        return False
    if base.endswith(".md"):
        return False
    lower = base.lower()
    if lower.endswith(".pt") or lower.endswith(".pth"):
        return True
    if "checkpoint" in lower:
        return True
    if "latent" in lower and (lower.endswith(".pkl") or lower.endswith(".csv")):
        return True
    return False


def collect_protected_exp_ids(top10_path: str, control_ids: Iterable[str] | None = None) -> Set[str]:
    protected = set(control_ids or [])
    if os.path.exists(_resolve_path(top10_path)):
        df = pd.read_csv(_resolve_path(top10_path))
        protected.update(df["ID"].astype(str).tolist())
    return protected


def plan_deletions(
    pretrain_dir: str,
    protected_exp_ids: Set[str],
    include_failed: bool = True,
) -> List[str]:
    planned = []
    pretrain_dir = _resolve_path(pretrain_dir)
    for exp_dir in sorted(glob(os.path.join(pretrain_dir, "exp_*"))):
        exp_id = os.path.basename(exp_dir)
        if exp_id in protected_exp_ids:
            continue
        for root, _, files in os.walk(exp_dir):
            for name in files:
                full = os.path.join(root, name)
                if _matches_large_pattern(full):
                    planned.append(full)
    if include_failed:
        return planned
    return planned


def apply_retention(
    run_dir: str,
    top10_path: str,
    apply: bool = False,
    log_path: str | None = None,
) -> pd.DataFrame:
    run_dir = _resolve_path(run_dir)
    pretrain_dir = os.path.join(run_dir, "pretrain")
    protected = collect_protected_exp_ids(top10_path)
    planned = plan_deletions(pretrain_dir, protected)

    rows = []
    for path in planned:
        action = "delete" if apply else "planned_delete"
        if apply and os.path.exists(path):
            os.remove(path)
        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": path,
                "action": action,
                "protected_exp": False,
            }
        )

    log_file = log_path or os.path.join(run_dir, "artifact_retention_log.csv")
    log_df = pd.DataFrame(rows)
    if os.path.exists(log_file):
        old = pd.read_csv(log_file)
        log_df = pd.concat([old, log_df], ignore_index=True)
    log_df.to_csv(log_file, index=False)
    return log_df


def main():
    parser = argparse.ArgumentParser("artifact_retention")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top10", required=True)
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default is dry-run)")
    args = parser.parse_args()
    df = apply_retention(args.run_dir, args.top10, apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] planned/processed deletions: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
