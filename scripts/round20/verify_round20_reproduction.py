#!/usr/bin/env python3
"""Verify Round 20 frozen ensemble and raw-forward capability."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.reproduction import run_reproduction_audit
from tools.round20.result_contracts import DEFAULT_RUN_ROOT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--release-dir", type=Path, default=DEFAULT_RUN_ROOT / "stage20e_release")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--mode", choices=["frozen", "raw", "both"], default="both")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()
    run_reproduction_audit(
        run_root=args.run_root,
        release_dir=args.release_dir,
        mode=args.mode,
        strict=args.strict,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()
