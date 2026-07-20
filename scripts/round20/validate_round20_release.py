#!/usr/bin/env python3
"""Validate Round 20 release directory integrity."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.release_integrity import validate_release_directory
from tools.round20.result_contracts import DEFAULT_RUN_ROOT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "stage20e_release",
    )
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    validate_release_directory(args.release_dir, strict=args.strict)


if __name__ == "__main__":
    main()
