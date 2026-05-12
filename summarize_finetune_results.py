"""
Standalone result aggregation tool for finetune runs.

Usage example:
python summarize_finetune_results.py \
    --result_root ./result/pretrain_vaewc_loss_v1
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SkipRecord:
    model_dir: str
    param_dir: str
    reason: str


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_param_value(value):
    if isinstance(value, list):
        return str(value)
    return value


def _collect_param_dirs(result_root: str, recursive_model_dir: bool) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if recursive_model_dir:
        for model_name in sorted(os.listdir(result_root)):
            model_path = os.path.join(result_root, model_name)
            if not os.path.isdir(model_path):
                continue
            if not model_name.startswith("exp_"):
                continue
            for param_name in sorted(os.listdir(model_path)):
                param_path = os.path.join(model_path, param_name)
                if os.path.isdir(param_path) and param_name.startswith("param_"):
                    pairs.append((model_path, param_path))
    else:
        for param_name in sorted(os.listdir(result_root)):
            param_path = os.path.join(result_root, param_name)
            if os.path.isdir(param_path) and param_name.startswith("param_"):
                pairs.append((result_root, param_path))
    return pairs


def _extract_params(
    row: Dict,
    params_used: Dict,
    config: Optional[Dict],
) -> Dict:
    output = {}
    if config:
        finetune_keys = config.get("finetune_params", {}).keys()
        classifier_keys = config.get("classifier_params", {}).keys()
        model_keys = config.get("model_params", {}).keys()
    else:
        finetune_keys = ("ftlr", "scheduler_flag", "loss_type", "focal_loss_gamma")
        classifier_keys = ("hidden_dims", "dropout_rate", "use_batch_norm", "activation")
        model_keys = ("gin_type", "mini_batch_size")

    for key in finetune_keys:
        if key in params_used:
            output[f"Finetune_{key.upper()}"] = _normalize_param_value(params_used[key])
        elif key in row:
            output[f"Finetune_{key.upper()}"] = _normalize_param_value(row[key])

    for key in classifier_keys:
        if key in params_used:
            output[f"Classifier_{key.upper()}"] = _normalize_param_value(params_used[key])
        elif key in row:
            output[f"Classifier_{key.upper()}"] = _normalize_param_value(row[key])

    for key in model_keys:
        if key in params_used:
            output[f"Model_{key.upper()}"] = _normalize_param_value(params_used[key])
        elif key in row:
            output[f"Model_{key.upper()}"] = _normalize_param_value(row[key])

    return output


def _build_row(
    item_id: int,
    model_dir: str,
    param_dir: str,
    metrics_row: Dict,
    params_used: Dict,
    config: Optional[Dict],
) -> Dict:
    out = {"ID": item_id}
    model_id = params_used.get("Model_ID", metrics_row.get("Model_ID", os.path.basename(model_dir)))
    out["Model_ID"] = model_id
    out["Model_Dir"] = os.path.basename(model_dir)
    out["Param_Dir"] = os.path.basename(param_dir)

    out.update(_extract_params(metrics_row, params_used, config))

    fixed_metric_keys = (
        "Train_AUC",
        "Val_AUC",
        "Test_AUC",
        "Train_AUPRC",
        "Val_AUPRC",
        "Test_AUPRC",
        "Best_Epoch",
        "Global_TCGA_AUC",
        "Global_TCGA_AUPRC",
        "Average_TCGA_AUC",
        "Average_TCGA_AUPRC",
        "TCGA2_Global_TCGA_AUC",
        "TCGA2_Global_TCGA_AUPRC",
        "TCGA2_Average_TCGA_AUC",
        "TCGA2_Average_TCGA_AUPRC",
    )
    for key in fixed_metric_keys:
        out[key] = metrics_row.get(key)

    for key, value in metrics_row.items():
        if "_TCGA_" in key and key not in out:
            out[key] = value

    return out


def _create_comparison_tables(all_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    if not all_rows:
        return [], []
    all_columns = []
    seen = set()
    for row in all_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_columns.append(key)

    individual_drug_cols = [
        c
        for c in all_columns
        if "_TCGA_" in c
        and not c.startswith("Global_")
        and not c.startswith("Average_")
        and not c.startswith("TCGA2_Global_")
        and not c.startswith("TCGA2_Average_")
    ]
    detailed_drop = set(individual_drug_cols)
    detailed_rows = [{k: v for k, v in row.items() if k not in detailed_drop} for row in all_rows]

    param_cols = [
        c
        for c in all_columns
        if (c.startswith("Finetune_") or c.startswith("Classifier_") or c.startswith("Model_"))
        and c != "Model_ID"
    ]
    focus_drop = set(param_cols)
    focus_rows = [{k: v for k, v in row.items() if k not in focus_drop} for row in all_rows]
    return detailed_rows, focus_rows


def _read_single_row_csv(path: str) -> Dict:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    raise ValueError("metrics_summary.csv is empty")


def _write_rows_csv(path: str, rows: List[Dict]) -> None:
    all_columns = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_columns.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_results(
    result_root: str,
    output_dir: str,
    recursive_model_dir: bool = True,
    strict: bool = False,
) -> Dict:
    config = None
    config_path = os.path.join(result_root, "finetune_config_used.json")
    if os.path.isfile(config_path):
        try:
            config = _load_json(config_path).get("config")
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            print(f"[WARN] Cannot parse config file: {config_path}. Error: {exc}")

    param_pairs = _collect_param_dirs(result_root, recursive_model_dir=recursive_model_dir)
    rows: List[Dict] = []
    skipped: List[SkipRecord] = []

    for model_dir, param_dir in param_pairs:
        metrics_path = os.path.join(param_dir, "metrics_summary", "metrics_summary.csv")
        params_path = os.path.join(param_dir, "params_used.json")

        if not os.path.isfile(metrics_path):
            skipped.append(SkipRecord(model_dir, param_dir, "missing_metrics_summary_csv"))
            continue

        try:
            metrics_row = _read_single_row_csv(metrics_path)
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            skipped.append(SkipRecord(model_dir, param_dir, f"invalid_metrics_csv: {exc}"))
            continue

        params_used = {}
        if os.path.isfile(params_path):
            try:
                params_used = _load_json(params_path)
            except Exception as exc:  # noqa: BLE001
                if strict:
                    raise
                skipped.append(SkipRecord(model_dir, param_dir, f"invalid_params_used_json: {exc}"))
                continue

        rows.append(_build_row(len(rows) + 1, model_dir, param_dir, metrics_row, params_used, config))

    if not rows:
        raise RuntimeError("No valid parameter results found. Nothing to aggregate.")

    detailed_rows, focus_rows = _create_comparison_tables(rows)

    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, "parameter_comparison_detailed.csv")
    focus_path = os.path.join(output_dir, "parameter_comparison_tcga_focus.csv")
    report_path = os.path.join(output_dir, "aggregation_report.json")

    _write_rows_csv(detailed_path, detailed_rows)
    _write_rows_csv(focus_path, focus_rows)

    report = {
        "result_root": os.path.abspath(result_root),
        "output_dir": os.path.abspath(output_dir),
        "scanned_param_dirs": len(param_pairs),
        "included_param_dirs": len(rows),
        "skipped_param_dirs": len(skipped),
        "skipped": [
            {
                "model_dir": os.path.basename(s.model_dir),
                "param_dir": os.path.basename(s.param_dir),
                "reason": s.reason,
            }
            for s in skipped
        ],
        "outputs": {
            "detailed_csv": detailed_path,
            "tcga_focus_csv": focus_path,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def parse_args():
    parser = argparse.ArgumentParser("summarize_finetune_results")
    parser.add_argument(
        "--result_root",
        type=str,
        required=True,
        help="Root result directory to scan (e.g. ./result/pretrain_vaewc_loss_v1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for summary CSVs/report. Default: result_root",
    )
    parser.add_argument(
        "--recursive_model_dir",
        action="store_true",
        default=True,
        help="Scan exp_*/param_* under result_root (default: enabled).",
    )
    parser.add_argument(
        "--no_recursive_model_dir",
        action="store_false",
        dest="recursive_model_dir",
        help="Disable recursive scan and only scan param_* directly under result_root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast instead of skipping malformed files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_root = args.result_root
    output_dir = args.output_dir or result_root
    summary = aggregate_results(
        result_root=result_root,
        output_dir=output_dir,
        recursive_model_dir=args.recursive_model_dir,
        strict=args.strict,
    )

    print("=" * 80)
    print("Aggregation complete")
    print(f"Scanned param dirs : {summary['scanned_param_dirs']}")
    print(f"Included param dirs: {summary['included_param_dirs']}")
    print(f"Skipped param dirs : {summary['skipped_param_dirs']}")
    print(f"Detailed CSV       : {summary['outputs']['detailed_csv']}")
    print(f"TCGA Focus CSV     : {summary['outputs']['tcga_focus_csv']}")
    print(f"Report JSON        : {os.path.join(output_dir, 'aggregation_report.json')}")
    print("=" * 80)
