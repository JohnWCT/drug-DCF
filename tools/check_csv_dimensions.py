#!/usr/bin/env python3
"""Inspect CSV dimensions quickly.

By default, this script checks:
1) /workspace/DAPL_git/data/0_Omics_old_table/pretrain_tcga.csv
2) /workspace/DAPL_git/data/0_Omics_old_table/pretrain_ccle.csv

Usage examples (run inside Docker container context):
    python3 tools/check_csv_dimensions.py
    python3 tools/check_csv_dimensions.py --csv /path/a.csv --csv /path/b.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


DEFAULT_CSVS = [
    "/workspace/DAPL_git/data/0_Omics_old_table/pretrain_tcga.csv",
    "/workspace/DAPL_git/data/0_Omics_old_table/pretrain_ccle.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        action="append",
        help="CSV path to inspect (can be provided multiple times).",
    )
    return parser.parse_args()


def resolve_targets(raw_csvs: List[str] | None) -> Iterable[Path]:
    targets = raw_csvs if raw_csvs else DEFAULT_CSVS
    return [Path(p).expanduser() for p in targets]


def inspect_one(csv_path: Path) -> Tuple[int, int]:
    df = pd.read_csv(csv_path)
    return df.shape


def main() -> None:
    args = parse_args()
    for csv_path in resolve_targets(args.csv):
        if not csv_path.exists():
            print(f"[missing] {csv_path}")
            continue
        rows, cols = inspect_one(csv_path)
        print(f"[ok] {csv_path} | shape=({rows}, {cols})")


if __name__ == "__main__":
    main()
