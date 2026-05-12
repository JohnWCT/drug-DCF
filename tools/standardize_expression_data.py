#!/usr/bin/env python3
"""Create z-score standardized expression CSVs for training/inference.

This script standardizes each gene column with:
    z = (x - mean) / (std + 1e-8)
and writes new CSV files while preserving sample IDs (index) and gene columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


EPS = 1e-8


def zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    return (df - mean) / (std + EPS)


def standardize_one(input_path: Path, output_path: Path) -> Tuple[int, int]:
    df = pd.read_csv(input_path, index_col=0)
    z_df = zscore_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z_df.to_csv(output_path)
    return z_df.shape


def parse_pairs(raw_pairs: List[str]) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for raw in raw_pairs:
        if "=>" not in raw:
            raise ValueError(f"Invalid pair format: {raw}. Expected input=>output")
        src, dst = raw.split("=>", 1)
        pairs.append((Path(src).expanduser(), Path(dst).expanduser()))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Input/output mapping in format: input.csv=>output_zscore.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = parse_pairs(args.pair)
    for src, dst in pairs:
        rows, cols = standardize_one(src, dst)
        print(f"[done] {src} -> {dst} | shape=({rows}, {cols})")


if __name__ == "__main__":
    main()
