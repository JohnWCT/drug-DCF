#!/usr/bin/env python3
"""
Run shell commands through RTK inside the DAPL container.

RTK intercepts CLI output and returns a compact summary, reducing tokens
when command results are fed to an LLM or logged for review.

Usage (inside container):
    python3 /workspace/DAPL/scripts/rtk_run.py git status
    python3 /workspace/DAPL/scripts/rtk_run.py --raw python3 -c "print('hi')"

From host:
    docker exec DAPL python3 /workspace/DAPL/scripts/rtk_run.py git status
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Sequence


def _resolve_rtk() -> str:
    """Prefer ~/.local/bin/rtk (install script default), then PATH."""
    local = os.path.expanduser("~/.local/bin/rtk")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    found = shutil.which("rtk")
    if found:
        return found
    raise FileNotFoundError(
        "rtk not found. Install first:\n"
        "  docker exec DAPL bash /workspace/DAPL/scripts/install_rtk.sh"
    )


def build_argv(command: Sequence[str], use_rtk: bool) -> List[str]:
    if not command:
        raise ValueError("command is required")
    if not use_rtk:
        return list(command)
    return [_resolve_rtk(), *command]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a command via RTK for token-efficient output."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip RTK and run the command directly.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after '--' or as trailing args (e.g. git status).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command")

    try:
        full = build_argv(cmd, use_rtk=not args.raw)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 127

    result = subprocess.run(full, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
