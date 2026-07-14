#!/usr/bin/env python3
"""Telegram notifications for Round 18 architecture screening stages."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.telegram_notify import send_telegram_message

DEFAULT_ROOT = Path("result/optimization_runs/round18_architecture")


def _root() -> Path:
    return Path(os.environ.get("ROUND18_ROOT", str(DEFAULT_ROOT)))


def _notify(text: str, *, fail_silently: bool = True) -> None:
    ok = send_telegram_message(text, fail_silently=fail_silently)
    if not ok:
        print("[round18_telegram_notify] skipped (Telegram not configured)", file=sys.stderr)


def _count_done(manifest: Path) -> Dict[str, int]:
    if not manifest.is_file():
        return {"total": 0, "done": 0, "failed": 0, "missing": 0}
    df = pd.read_csv(manifest)
    done = failed = missing = 0
    for _, row in df.iterrows():
        st = Path(str(row["result_dir"])) / "job_status.json"
        if not st.is_file():
            missing += 1
            continue
        status = json.loads(st.read_text()).get("status")
        if status == "done":
            done += 1
        else:
            failed += 1
    return {"total": len(df), "done": done, "failed": failed, "missing": missing}


def notify_stage_start(stage: str, extra: str = "") -> None:
    lines = [f"[Round 18] Stage {stage} 開始"]
    if extra:
        lines.append(extra)
    _notify("\n".join(lines))


def notify_stage_done(stage: str, manifest: Optional[str] = None) -> None:
    lines = [f"[Round 18] Stage {stage} 完成"]
    if manifest:
        stats = _count_done(Path(manifest))
        lines.append(
            f"jobs: done={stats['done']}/{stats['total']} "
            f"failed={stats['failed']} missing={stats['missing']}"
        )
    ranking = _root() / "reports" / "round18_screening_architecture_ranking.csv"
    if ranking.is_file():
        df = pd.read_csv(ranking)
        if not df.empty:
            top = df.iloc[0]
            lines.append(
                f"screening top: {top.get('architecture_id')} "
                f"AUC={float(top.get('mean_DrugMacro_AUC')):.4f}"
            )
    _notify("\n".join(lines))


def notify_stage_fail(stage: str, reason: str) -> None:
    _notify(f"[Round 18] Stage {stage} 失敗\nreason: {reason}")


def notify_lock_written(lock_path: Optional[str] = None) -> None:
    path = Path(lock_path or (_root() / "reports" / "round18_locked_selection.json"))
    lines = ["[Round 18] Selection lock 已寫入"]
    if path.is_file():
        lock = json.loads(path.read_text())
        lines.append(f"policy: {lock.get('selection_policy')}")
        lines.append(f"18B: {lock.get('stage18b_completion')}")
        lines.append(f"18C-A: {lock.get('stage18c_completion')}")
        lines.append(f"18C-B: {lock.get('stage18c_none_followup_completion')}")
        for c in lock.get("formal_candidates") or []:
            lines.append(
                f"- {c.get('role')}: {c.get('architecture_id')} "
                f"AUC={c.get('mean_DrugMacro_AUC')}"
            )
    _notify("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 18 Telegram notifications")
    parser.add_argument(
        "--event",
        required=True,
        choices=["stage-start", "stage-done", "stage-fail", "lock-written"],
    )
    parser.add_argument("--stage", default="")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--reason", default="")
    parser.add_argument("--extra", default="")
    parser.add_argument("--lock", default=None)
    args = parser.parse_args()
    if args.event == "stage-start":
        notify_stage_start(args.stage or "?", args.extra)
    elif args.event == "stage-done":
        notify_stage_done(args.stage or "?", args.manifest)
    elif args.event == "stage-fail":
        notify_stage_fail(args.stage or "?", args.reason or "unknown")
    elif args.event == "lock-written":
        notify_lock_written(args.lock)


if __name__ == "__main__":
    main()
