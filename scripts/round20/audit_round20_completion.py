#!/usr/bin/env python3
"""Round 20 completion audit CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.completion_audit import run_completion_audit
from tools.round20.predictor_contract import build_gate_summary, build_predictor_contract
from tools.round20.result_contracts import DEFAULT_RUN_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description="Audit Round 20 stage completion")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()
    build_predictor_contract(run_root=args.run_root)
    build_gate_summary(run_root=args.run_root)
    audit = run_completion_audit(
        run_root=args.run_root,
        strict=args.strict,
        write_artifacts=not args.no_write,
    )
    print(f"ROUND20_COMPLETION_AUDIT={audit['audit_status']}")
    if audit.get("blocking_errors"):
        print("blocking:", audit["blocking_errors"])
    if audit.get("warnings"):
        print("warnings:", audit["warnings"])


if __name__ == "__main__":
    main()
