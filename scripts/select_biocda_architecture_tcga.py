#!/usr/bin/env python3
"""Select BioCDA architecture by TCGA target priority (no GDSC test)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.validation.tcga_architecture_selection import write_selection_artifacts
from tools.biocda_telegram_notify import biocda_notify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports/biocda_tcga_architecture_selection.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "docs/biocda_final_architecture_selection.md",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing TCGA comparison CSV: {args.input}", file=sys.stderr)
        print("Run: python3 scripts/compare_biocda_tcga.py", file=sys.stderr)
        return 1

    payload = write_selection_artifacts(
        long_csv=args.input,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    winner = payload["winner_weighted"]
    biocda_notify(
        f"BioCDA TCGA architecture selection DONE\n"
        f"winner={winner['display_name']}\n"
        f"weighted DrugMacro AUC={winner['weighted_DrugMacro_AUC']:.4f}"
    )
    print(json.dumps(payload["final_recommendation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
