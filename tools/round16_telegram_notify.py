#!/usr/bin/env python3
"""Telegram notifications for Round 16 pipeline stages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.telegram_notify import send_telegram_message

ROUND16_ROOT = Path("result/optimization_runs/round16_bruteforce")
MANIFESTS: Dict[str, Path] = {
    "16F": ROUND16_ROOT / "manifests/stage16f_finetune_dispatch_manifest.csv",
    "16E": ROUND16_ROOT / "manifests/stage16e_finetune_dispatch_manifest.csv",
    "16D": ROUND16_ROOT / "stage16d/manifests/stage16d_pretrain_manifest.csv",
    "16A": ROUND16_ROOT / "manifests/finetune_dispatch_manifest.csv",
    "16B": ROUND16_ROOT / "manifests/stage16b_finetune_dispatch_manifest.csv",
    "16C": ROUND16_ROOT / "manifests/stage16c_finetune_dispatch_manifest.csv",
}
EXPECTED_JOBS: Dict[str, int] = {
    "16F": 384,
    "16E": 432,
    "16D": 60,
    "16A": 1152,
    "16B": 100,
    "16C": 432,
}


def _summarize_manifest(path: Path) -> Dict[str, int]:
    if not path.is_file():
        return {"total": 0, "success": 0, "failed": 0, "running": 0, "pending": 0}
    df = pd.read_csv(path)
    total = len(df)
    if "status" not in df.columns:
        return {"total": total, "success": 0, "failed": 0, "running": 0, "pending": total}
    vc = df["status"].value_counts()
    return {
        "total": total,
        "success": int(vc.get("success", 0)),
        "failed": int(vc.get("failed", 0)),
        "running": int(vc.get("running", 0)),
        "pending": int(vc.get("pending", 0)),
    }


def reset_manifests() -> int:
    reset = 0
    for path in MANIFESTS.values():
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "status" not in df.columns:
            continue
        df["status"] = "pending"
        for col in ("start_time", "end_time", "error_message"):
            if col in df.columns:
                df[col] = ""
        df.to_csv(path, index=False)
        reset += len(df)
    return reset


def _notify(text: str, *, fail_silently: bool = True) -> None:
    ok = send_telegram_message(text, fail_silently=fail_silently)
    if not ok:
        print("[round16_telegram_notify] skipped (Telegram not configured)", file=sys.stderr)


def notify_pipeline_start(stages: Optional[str] = None) -> None:
    order = stages or os.environ.get("ROUND16_PIPELINE_STAGES", "16f,16e,16d")
    order = order.replace(",", " → ").upper()
    deferred = Path("config/round16_defer_downstream.flag").is_file()
    defer_note = "\n16A–16C 延後至下個 round" if deferred else ""
    _notify(
        f"[Round 16] Pipeline 開始\n"
        f"順序: {order}\n"
        f"16D = pretrain VICReg micro-search{defer_note}"
    )


def notify_pipeline_done(stages: Optional[str] = None) -> None:
    stage_list = [
        s.strip().upper()
        for s in (stages or os.environ.get("ROUND16_PIPELINE_STAGES", "16f,16e,16d")).split(",")
        if s.strip()
    ]
    lines = ["[Round 16] Pipeline 完成", f"Stages: {', '.join(stage_list)}"]
    for stage in stage_list:
        path = MANIFESTS.get(stage)
        stats = _summarize_manifest(path) if path else {"total": 0, "success": 0, "failed": 0}
        if stats["total"] == 0:
            lines.append(f"{stage}: 未執行 / 無 manifest")
        else:
            lines.append(
                f"{stage}: 成功 {stats['success']} | 失敗 {stats['failed']} | 總計 {stats['total']}"
            )
    _notify("\n".join(lines))


def notify_stage_start(stage: str) -> None:
    stage = stage.upper()
    expected = EXPECTED_JOBS.get(stage, "?")
    path = MANIFESTS.get(stage)
    stats = _summarize_manifest(path) if path else {}
    lines = [f"[Round 16] Stage {stage} 開始", f"預期 jobs: {expected}"]
    if stats.get("total", 0) > 0:
        lines.append(
            f"目前: 成功 {stats.get('success', 0)} | 失敗 {stats.get('failed', 0)} | "
            f"執行中 {stats.get('running', 0)} | 待跑 {stats.get('pending', 0)}"
        )
    _notify("\n".join(lines))


def notify_stage_done(stage: str, manifest: Optional[Path] = None) -> None:
    stage = stage.upper()
    path = manifest or MANIFESTS.get(stage)
    stats = _summarize_manifest(path) if path else {"total": 0, "success": 0, "failed": 0}
    expected = EXPECTED_JOBS.get(stage, stats["total"])
    status = "完成"
    if stats["failed"] > 0:
        status = "完成（有失敗）"
    elif stats["success"] < expected:
        status = "結束（未跑滿）"
    _notify(
        f"[Round 16] Stage {stage} {status}\n"
        f"成功: {stats['success']} | 失敗: {stats['failed']} | 總計: {stats['total']}"
    )


def notify_stage_fail(stage: str, reason: str) -> None:
    stage = stage.upper()
    path = MANIFESTS.get(stage)
    stats = _summarize_manifest(path) if path else {"total": 0, "success": 0, "failed": 0}
    _notify(
        f"[Round 16] Stage {stage} 失敗\n"
        f"原因: {reason}\n"
        f"目前: 成功 {stats['success']} | 失敗 {stats['failed']} | 總計 {stats['total']}"
    )


def notify_pipeline_fail(reason: str) -> None:
    lines = [f"[Round 16] Pipeline 中斷\n原因: {reason}"]
    for stage in ("16F", "16E", "16D", "16A", "16B", "16C"):
        stats = _summarize_manifest(MANIFESTS[stage])
        if stats["total"] > 0:
            lines.append(
                f"{stage}: 成功 {stats['success']} | 失敗 {stats['failed']} | 總計 {stats['total']}"
            )
    _notify("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Round 16 Telegram notifications")
    parser.add_argument(
        "--event",
        required=True,
        choices=(
            "pipeline-start",
            "pipeline-done",
            "pipeline-fail",
            "stage-start",
            "stage-done",
            "stage-fail",
            "reset-manifests",
        ),
    )
    parser.add_argument("--stage", default=None, help="Stage id, e.g. 16F")
    parser.add_argument("--stages", default=None, help="Comma-separated pipeline stages for start/done events")
    parser.add_argument("--reason", default="", help="Failure reason")
    parser.add_argument("--manifest", default=None, help="Optional manifest path for stage-done")
    args = parser.parse_args()

    if args.event == "reset-manifests":
        n = reset_manifests()
        print(f"Reset {n} manifest rows to pending.")
        return 0
    if args.event == "pipeline-start":
        notify_pipeline_start(args.stages)
    elif args.event == "pipeline-done":
        notify_pipeline_done(args.stages)
    elif args.event == "pipeline-fail":
        notify_pipeline_fail(args.reason or "unknown error")
    elif args.event == "stage-start":
        if not args.stage:
            raise SystemExit("--stage is required")
        notify_stage_start(args.stage)
    elif args.event == "stage-done":
        if not args.stage:
            raise SystemExit("--stage is required")
        manifest = Path(args.manifest) if args.manifest else None
        notify_stage_done(args.stage, manifest)
    elif args.event == "stage-fail":
        if not args.stage:
            raise SystemExit("--stage is required")
        notify_stage_fail(args.stage, args.reason or "unknown error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
