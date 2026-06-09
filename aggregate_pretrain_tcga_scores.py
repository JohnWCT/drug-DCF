"""
Aggregate finetune TCGA scores by pretrain Model_ID.

Reads parameter_comparison_tcga_focus.csv (one row per finetune param combo),
averages numeric prediction metrics per Model_ID, and ranks pretrain models by
mean Global_TCGA_AUC. Stability columns (std / range) help compare robustness
across finetune hyperparameters.

Usage:
  docker exec DAPL python3 /workspace/DAPL/aggregate_pretrain_tcga_scores.py

  docker exec DAPL python3 /workspace/DAPL/aggregate_pretrain_tcga_scores.py \
    --input result/pretrain_vaewc_loss_v2/parameter_comparison_tcga_focus.csv \
    --output result/pretrain_vaewc_loss_v2/pretrain_tcga_model_summary.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


KEY_STABILITY_METRICS = [
    "Global_TCGA_AUC",
    "Global_TCGA_AUPRC",
    "Average_TCGA_AUC",
    "Average_TCGA_AUPRC",
    "Test_AUC",
    "Test_AUPRC",
    "Val_AUC",
    "TCGA2_Global_TCGA_AUC",
    "TCGA2_Average_TCGA_AUC",
]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"ID", "Model_ID"}
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if "Model_ID" not in df.columns:
        raise ValueError("Input CSV must contain a Model_ID column")

    work = df.copy()
    work["Model_ID"] = work["Model_ID"].astype(str)
    numeric_cols = _numeric_columns(work)

    grouped = work.groupby("Model_ID", sort=False)
    mean_df = grouped[numeric_cols].mean()
    std_df = grouped[numeric_cols].std(ddof=0)
    count = grouped.size().rename("n_finetune_runs")

    out = mean_df.add_suffix("_mean")
    out = out.join(std_df.add_suffix("_std"))
    out = out.join(count)

    for metric in KEY_STABILITY_METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in out.columns and std_col in out.columns:
            out[f"{metric}_cv"] = out[std_col] / out[mean_col].replace(0, pd.NA)

    sort_col = "Global_TCGA_AUC_mean"
    if sort_col not in out.columns:
        raise ValueError(f"Missing required column after aggregation: {sort_col}")

    out = out.sort_values(sort_col, ascending=False, na_position="last")
    out.insert(0, "rank_by_Global_TCGA_AUC", range(1, len(out) + 1))
    out = out.reset_index()
    return out


def _print_top_summary(df: pd.DataFrame, top_n: int = 10) -> None:
    cols = [
        "rank_by_Global_TCGA_AUC",
        "Model_ID",
        "n_finetune_runs",
        "Global_TCGA_AUC_mean",
        "Global_TCGA_AUC_std",
        "Average_TCGA_AUC_mean",
        "Test_AUC_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(top_n).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average finetune TCGA scores by pretrain Model_ID."
    )
    parser.add_argument(
        "--input",
        default="result/pretrain_vaewc_loss_v2/parameter_comparison_tcga_focus.csv",
        help="Finetune comparison CSV from step1_finetune pipeline",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input_dir>/pretrain_tcga_model_summary.csv)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top models to print",
    )
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    raw = pd.read_csv(input_path)
    if raw.empty:
        raise ValueError(f"No rows in input: {args.input}")

    summary = aggregate_by_model(raw)

    if args.output is None:
        out_dir = os.path.dirname(input_path)
        out_path = os.path.join(out_dir, "pretrain_tcga_model_summary.csv")
    else:
        out_path = _resolve_path(args.output)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(f"Input rows: {len(raw)} ({raw['Model_ID'].nunique()} Model_ID)")
    print(f"Saved: {out_path}")
    print()
    print(f"Top {args.top_n} pretrain models by mean Global_TCGA_AUC:")
    _print_top_summary(summary, top_n=args.top_n)


if __name__ == "__main__":
    main()
