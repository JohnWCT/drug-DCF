#!/usr/bin/env python3
"""Minimal demo: integrate Telegram notify into a long-running job."""

from __future__ import annotations

import sys
import time

# Allow running as: python3 scripts/telegram_demo.py
sys.path.insert(0, "/workspace/DAPL")

from tools.telegram_notify import send_telegram_message


def main() -> int:
    job_name = "DAPL demo job"
    send_telegram_message(
        f"[開始] {job_name}",
        fail_silently=False,
    )

    # Simulate work
    time.sleep(2)

    send_telegram_message(
        f"[完成] {job_name}\n耗時約 2 秒",
        fail_silently=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
