#!/usr/bin/env python3
"""Telegram notifications for Round 19 factorial stages."""
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

DEFAULT_ROOT = Path("result/optimization_runs/round19_factorial")


def _root() -> Path:
    return Path(os.environ.get("ROUND19_ROOT", str(DEFAULT_ROOT)))


def _notify(text: str, *, fail_silently: bool = True) -> None:
    ok = send_telegram_message(text, fail_silently=fail_silently)
    if not ok:
        print("[round19_telegram_notify] skipped (Telegram not configured)", file=sys.stderr)


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
        status = json.loads(st.read_text(encoding="utf-8")).get("status")
        if status == "done":
            done += 1
        else:
            failed += 1
    return {"total": int(len(df)), "done": done, "failed": failed, "missing": missing}


def notify_stage_start(stage: str, extra: str = "") -> None:
    lines = [f"[Round 19] Stage {stage} 開始"]
    if extra:
        lines.append(extra)
    _notify("\n".join(lines))


def notify_stage_done(stage: str, manifest: Optional[str] = None) -> None:
    lines = [f"[Round 19] Stage {stage} 完成"]
    if manifest:
        stats = _count_done(Path(manifest))
        lines.append(
            f"jobs: done={stats['done']}/{stats['total']} "
            f"failed={stats['failed']} missing={stats['missing']}"
        )
    lock = _root() / "reports" / "round19_stage19c_candidate_lock.json"
    if stage.startswith("19c") and lock.is_file():
        payload = json.loads(lock.read_text(encoding="utf-8"))
        n = len(payload.get("unique_cells") or payload.get("selected_cells") or [])
        lines.append(f"selected_cells={n}")
    _notify("\n".join(lines))


def notify_stage_fail(stage: str, reason: str) -> None:
    _notify(f"[Round 19] Stage {stage} 失敗\nreason: {reason}")


def notify_progress(stage: str, manifest: Optional[str] = None) -> None:
    lines = [f"[Round 19] Stage {stage} 進度"]
    if manifest:
        stats = _count_done(Path(manifest))
        lines.append(
            f"jobs: done={stats['done']}/{stats['total']} "
            f"failed={stats['failed']} missing={stats['missing']}"
        )
    _notify("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 Telegram notifications")
    parser.add_argument(
        "--event",
        required=True,
        choices=["stage-start", "stage-done", "stage-fail", "progress"],
    )
    parser.add_argument("--stage", default="19c")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--extra", default="")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.event == "stage-start":
        notify_stage_start(args.stage, args.extra)
    elif args.event == "stage-done":
        notify_stage_done(args.stage, args.manifest)
    elif args.event == "stage-fail":
        notify_stage_fail(args.stage, args.reason or "unknown")
    elif args.event == "progress":
        notify_progress(args.stage, args.manifest)


if __name__ == "__main__":
    main()
