#!/usr/bin/env python3
"""Telegram notifications for Round 17 direct-prototype pipeline stages."""

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

DEFAULT_ROOT = Path("result/optimization_runs/round17_direct_proto")


def _round17_root() -> Path:
    return Path(os.environ.get("ROUND17R_ROOT") or os.environ.get("ROUND17_ROOT", str(DEFAULT_ROOT)))


def _prefix(stage: Optional[str] = None) -> str:
    if stage and str(stage).upper().startswith("17R"):
        return "[Round 17R]"
    return "[Round 17]"


def _manifests(root: Path) -> Dict[str, Path]:
    manifests = root / "manifests"
    return {
        "17A": manifests / "stage17a_finetune_dispatch_manifest.csv",
        "17B": manifests / "stage17b_finetune_dispatch_manifest.csv",
        "17C": manifests / "stage17c_finetune_dispatch_manifest.csv",
        "17R-A": manifests / "stage17r_a_proto_feature_manifest.csv",
        "17R-B": manifests / "stage17r_b_finetune_dispatch_manifest.csv",
        "17R-C": manifests / "stage17r_c_finetune_dispatch_manifest.csv",
        "17R-D": manifests / "stage17r_d_finetune_dispatch_manifest.csv",
    }


def _summarize_manifest(path: Optional[Path]) -> Dict[str, int]:
    if path is None or not path.is_file():
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


def _notify(text: str, *, fail_silently: bool = True) -> None:
    ok = send_telegram_message(text, fail_silently=fail_silently)
    if not ok:
        print("[round17_telegram_notify] skipped (Telegram not configured)", file=sys.stderr)


def notify_pipeline_start(stages: Optional[str] = None) -> None:
    order = stages or os.environ.get("ROUND17_PIPELINE_STAGES", "17a,17b,17c")
    order = order.replace(",", " → ").upper()
    parallel = os.environ.get("FINETUNE_PARALLEL", "?")
    batch = os.environ.get("FINETUNE_BATCH_SIZE", "?")
    mini = os.environ.get("FINETUNE_MINI_BATCH_SIZE", "?")
    _notify(
        f"[Round 17] Pipeline 開始\n"
        f"順序: {order}\n"
        f"GPU tuning: parallel={parallel} batch={batch} mini={mini}"
    )


def notify_pipeline_done(stages: Optional[str] = None) -> None:
    root = _round17_root()
    stage_list = [
        s.strip().upper()
        for s in (stages or os.environ.get("ROUND17_PIPELINE_STAGES", "17a,17b,17c")).split(",")
        if s.strip()
    ]
    manifests = _manifests(root)
    lines = ["[Round 17] Pipeline 完成", f"Stages: {', '.join(stage_list)}"]
    for stage in stage_list:
        path = manifests.get(stage)
        stats = _summarize_manifest(path)
        if stats["total"] == 0:
            lines.append(f"{stage}: 未執行 / 無 manifest")
        else:
            lines.append(
                f"{stage}: 成功 {stats['success']} | 失敗 {stats['failed']} | 總計 {stats['total']}"
            )
    _notify("\n".join(lines))


def notify_stage_start(stage: str) -> None:
    stage = stage.upper()
    root = _round17_root()
    path = _manifests(root).get(stage)
    stats = _summarize_manifest(path)
    lines = [f"{_prefix(stage)} Stage {stage} 開始"]
    if stats["total"] > 0:
        lines.append(f"manifest jobs: {stats['total']}")
        lines.append(
            f"目前: 成功 {stats.get('success', 0)} | 失敗 {stats.get('failed', 0)} | "
            f"執行中 {stats.get('running', 0)} | 待跑 {stats.get('pending', 0)}"
        )
    _notify("\n".join(lines))


def notify_stage_done(stage: str, manifest: Optional[Path] = None) -> None:
    stage = stage.upper()
    root = _round17_root()
    path = manifest or _manifests(root).get(stage)
    stats = _summarize_manifest(path)
    status = "完成"
    if stats["failed"] > 0:
        status = "完成（有失敗）"
    elif stats["success"] < stats["total"]:
        status = "結束（未跑滿）"
    _notify(
        f"{_prefix(stage)} Stage {stage} {status}\n"
        f"成功: {stats['success']} | 失敗: {stats['failed']} | 總計: {stats['total']}"
    )


def notify_stage17f_done(outdir: Optional[Path] = None) -> None:
    root = _round17_root()
    viz = outdir or (root / "visualizations" / "prototype_tsne")
    models: list[str] = []
    if viz.is_dir():
        for child in sorted(viz.iterdir()):
            if child.is_dir() and (child / "prototype_tsne_coordinates.csv").is_file():
                models.append(child.name)
    lines = ["[Round 17] Stage 17F 完成", f"輸出: {viz}", f"models: {len(models)}"]
    if models:
        lines.append(", ".join(models))
    else:
        lines.append("（未找到 tSNE 座標檔）")
    _notify("\n".join(lines))


def notify_stage_fail(stage: str, reason: str) -> None:
    stage = stage.upper()
    root = _round17_root()
    path = _manifests(root).get(stage)
    stats = _summarize_manifest(path)
    _notify(
        f"{_prefix(stage)} Stage {stage} 失敗\n"
        f"原因: {reason}\n"
        f"目前: 成功 {stats['success']} | 失敗 {stats['failed']} | 總計: {stats['total']}"
    )


def notify_pipeline_fail(reason: str) -> None:
    root = _round17_root()
    lines = [f"[Round 17] Pipeline 中斷\n原因: {reason}"]
    for stage, path in _manifests(root).items():
        stats = _summarize_manifest(path)
        if stats["total"] > 0:
            lines.append(
                f"{stage}: 成功 {stats['success']} | 失敗 {stats['failed']} | 總計 {stats['total']}"
            )
    _notify("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Round 17 Telegram notifications")
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
            "stage17f-done",
        ),
    )
    parser.add_argument("--stage", default=None, help="Stage id, e.g. 17A")
    parser.add_argument("--stages", default=None, help="Comma-separated pipeline stages for start/done events")
    parser.add_argument("--reason", default="", help="Failure reason")
    parser.add_argument("--manifest", default=None, help="Optional manifest path for stage-done")
    parser.add_argument("--outdir", default=None, help="Optional Stage 17F visualization outdir")
    args = parser.parse_args()

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
    elif args.event == "stage17f-done":
        outdir = Path(args.outdir) if args.outdir else None
        notify_stage17f_done(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
