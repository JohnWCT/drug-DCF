#!/usr/bin/env python3
"""Generate Round 20 documentation from locked artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.documentation_builder import generate_all_docs
from tools.round20.result_contracts import DEFAULT_RUN_ROOT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    audit_path = args.run_root / "round20_completion_audit.json"
    if args.strict and not audit_path.is_file():
        raise SystemExit("completion audit missing; run audit_round20_completion.py first")
    paths = generate_all_docs(run_root=args.run_root, docs_dir=args.docs_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
