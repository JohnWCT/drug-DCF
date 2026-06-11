"""Collect integrated TCGA eval summaries from finetune result folders."""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def find_integrated_summaries(roots: List[str]) -> pd.DataFrame:
    rows = []
    for root in roots:
        root = _resolve(root)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "eval_metrics_integrated_summary.csv" not in filenames:
                continue
            path = os.path.join(dirpath, "eval_metrics_integrated_summary.csv")
            part = pd.read_csv(path)
            if part.empty:
                continue
            part = part.copy()
            part["source_run_dir"] = dirpath
            rel = os.path.relpath(dirpath, root)
            part["source_root"] = root
            part["source_rel"] = rel
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ("Integrated_Global_TCGA_AUC", "Global_TCGA_AUC") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols[0], ascending=False, na_position="last")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect integrated TCGA eval CSVs")
    parser.add_argument(
        "roots",
        nargs="+",
        help="Result root folders to scan (e.g. result/pretrain_vaewc_loss)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="integrated_tcga_eval_collection.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    df = find_integrated_summaries(args.roots)
    if df.empty:
        print("No eval_metrics_integrated_summary.csv found.")
        return
    out_path = _resolve(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Collected {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
