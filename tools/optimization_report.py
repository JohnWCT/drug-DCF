"""Generate final CSV/JSON/Markdown reports for optimization runs."""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_DOWNSTREAM_METRIC = "Average_TCGA_AUC_mean"


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    path = _resolve_path(path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def generate_final_reports(run_dir: str) -> dict:
    run_dir = _resolve_path(run_dir)
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    aggregate_df = _safe_read_csv(os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    pretrain_manifest = _safe_read_csv(os.path.join(run_dir, "manifests", "pretrain_sweep_manifest.csv"))
    finetune_manifest = _safe_read_csv(os.path.join(run_dir, "manifests", "finetune_dispatch_manifest.csv"))
    top10_df = _safe_read_csv(os.path.join(run_dir, "selection", "pretrain_top10.csv"))
    filtered_df = _safe_read_csv(os.path.join(run_dir, "selection", "pretrain_filtered_candidates.csv"))

    summary = {
        "run_dir": run_dir,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "pretrain": {},
        "finetune": {},
        "downstream": {},
    }

    if pretrain_manifest is not None:
        summary["pretrain"] = {
            "total_jobs": int(len(pretrain_manifest)),
            "success": int((pretrain_manifest["status"] == "success").sum()),
            "failed": int((pretrain_manifest["status"] == "failed").sum()),
            "pending": int((pretrain_manifest["status"] == "pending").sum()),
        }

    if finetune_manifest is not None:
        summary["finetune"] = {
            "total_jobs": int(len(finetune_manifest)),
            "success": int((finetune_manifest["status"] == "success").sum()),
            "failed": int((finetune_manifest["status"] == "failed").sum()),
        }

    control_compare = {}
    if filtered_df is not None and "lambda_proto" in filtered_df.columns:
        controls = filtered_df[filtered_df["lambda_proto"].fillna(0.0) == 0.0]
        infonce = filtered_df[filtered_df["lambda_proto"].fillna(0.0) != 0.0]
        for col in ["score_total", "score_deconfounding", "fid", "mmd", "kmeans_ari"]:
            if col in filtered_df.columns:
                control_compare[col] = {
                    "control_mean": float(controls[col].mean()) if len(controls) else None,
                    "infonce_mean": float(infonce[col].mean()) if len(infonce) else None,
                }
    summary["control_vs_infonce"] = control_compare

    if aggregate_df is not None and not aggregate_df.empty:
        sort_col = (
            PRIMARY_DOWNSTREAM_METRIC
            if PRIMARY_DOWNSTREAM_METRIC in aggregate_df.columns
            else "Global_TCGA_AUC_mean"
            if "Global_TCGA_AUC_mean" in aggregate_df.columns
            else aggregate_df.columns[0]
        )
        ranked = aggregate_df.sort_values(sort_col, ascending=False, na_position="last")
        best_row = ranked.iloc[0]
        best_model_id = str(best_row.get("Model_ID", ranked.index[0]))
        summary["downstream"] = {
            "primary_metric": sort_col,
            "best_model_id": best_model_id,
            "top_candidates": ranked.head(10).reset_index().to_dict(orient="records"),
        }

    summary_path = os.path.join(reports_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    lines = [
        "# Final Selection Report",
        "",
        "## Environment",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        "",
        "## Pretrain sweep status",
    ]
    if pretrain_manifest is not None:
        lines.append(f"- Total jobs: {summary['pretrain'].get('total_jobs', 0)}")
        lines.append(f"- Success: {summary['pretrain'].get('success', 0)}")
        lines.append(f"- Failed: {summary['pretrain'].get('failed', 0)}")
    else:
        lines.append("- Pretrain manifest not found.")

    lines.extend(["", "## Top-10 pretrain selection"])
    if top10_df is not None:
        for _, row in top10_df.iterrows():
            tag = " [control]" if row.get("is_control") else ""
            lines.append(f"- {row['ID']}: score_total={row.get('score_total', 'NA')}{tag}")
    else:
        lines.append("- Top-10 file not found.")

    lines.extend(["", "## Control vs InfoNCE (pretrain metrics)"])
    for metric, values in control_compare.items():
        lines.append(
            f"- {metric}: control_mean={values.get('control_mean')} infonce_mean={values.get('infonce_mean')}"
        )

    lines.extend(["", f"## Downstream ranking ({PRIMARY_DOWNSTREAM_METRIC})"])
    if aggregate_df is not None and not aggregate_df.empty:
        sort_col = (
            PRIMARY_DOWNSTREAM_METRIC
            if PRIMARY_DOWNSTREAM_METRIC in aggregate_df.columns
            else "Global_TCGA_AUC_mean"
            if "Global_TCGA_AUC_mean" in aggregate_df.columns
            else aggregate_df.columns[0]
        )
        ranked = aggregate_df.sort_values(sort_col, ascending=False, na_position="last").head(10)
        for _, row in ranked.iterrows():
            model_id = row.get("Model_ID", row.name)
            global_auc = row.get("Global_TCGA_AUC_mean", "NA")
            lines.append(
                f"- {model_id}: {sort_col}={row.get(sort_col, 'NA')} "
                f"(Global_TCGA_AUC_mean={global_auc})"
            )
    else:
        lines.append("- Aggregate scores not available.")

    lines.extend(
        [
            "",
            "## Commands",
            "- Config generation: `python3 tools/optimization_config_generator.py`",
            "- Pretrain queue: `python3 tools/optimization_runner.py pretrain --manifest <manifest> --run-dir <run_dir>`",
            "- Selection: `python3 tools/optimization_runner.py select --run-dir <run_dir>`",
            "- Finetune: `python3 tools/optimization_runner.py finetune --manifest <ft_manifest> --run-dir <run_dir> --top10 <top10.csv>`",
            "- Aggregate: `python3 tools/optimization_runner.py aggregate --run-dir <run_dir>`",
            "- Report: `python3 tools/optimization_runner.py report --run-dir <run_dir>`",
        ]
    )

    report_path = os.path.join(reports_dir, "final_selection_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"summary_path": summary_path, "report_path": report_path}


def main():
    import argparse

    parser = argparse.ArgumentParser("optimization_report")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    outputs = generate_final_reports(args.run_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
