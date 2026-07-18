#!/usr/bin/env python3
"""CLI: Stage 20A paired analysis + dimension decision."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20_stage20a import analyze_stage20a  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default="result/optimization_runs/round20_unseen_drug_closure/stage20a_dimension",
    )
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--no-strict", action="store_true")
    args = p.parse_args()
    decision = analyze_stage20a(
        stage_dir=Path(args.input_dir),
        strict=not args.no_strict,
    )
    print(json.dumps(decision, indent=2))
    if decision.get("status") != "LOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
