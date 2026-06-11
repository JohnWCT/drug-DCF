"""TCGA eval targets and CODEAE-style finetune export helpers."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tools.dataprocess import safemakedirs

# Primary TCGA eval (GDSC-overlapping 13 drugs); replaces legacy intersect_pretrain for finetune eval.
FIXED_TCGA_EVAL_GDSC_INTERSECT13 = (
    "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv"
)
FIXED_TCGA_EVAL_TCGA_ONLY3 = (
    "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv"
)
FIXED_TCGA_EVAL_DAPL = "data/TCGA/TCGA_drug_response_from_DAPL.csv"

# Backward-compatible aliases used across finetune scripts.
FIXED_TCGA_DATA_FOLDER = FIXED_TCGA_EVAL_GDSC_INTERSECT13
FIXED_TCGA_DATA_FOLDER_EXTRA = FIXED_TCGA_EVAL_DAPL

DEFAULT_TCGA_EVAL_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("gdsc_intersect13", FIXED_TCGA_EVAL_GDSC_INTERSECT13),
    ("tcga_only3", FIXED_TCGA_EVAL_TCGA_ONLY3),
    ("dapl", FIXED_TCGA_EVAL_DAPL),
)

# Legacy tag mapping for prediction rows.
EVAL_KEY_TO_LEGACY_TAG = {
    "gdsc_intersect13": "TCGA1",
    "tcga_only3": "tcga_only3",
    "dapl": "TCGA2",
}


def load_tcga_response_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "drug_name" not in df.columns:
        if "drug.name" in df.columns:
            df = df.copy()
            df["drug_name"] = df["drug.name"]
        else:
            raise ValueError(f"TCGA CSV missing drug_name column: {path}")
    if "Patient_id" not in df.columns and "patient" in df.columns:
        df = df.copy()
        df["Patient_id"] = df["patient"]
    return df


def build_drug_list_df(drug_smiles_df: pd.DataFrame, response_df: pd.DataFrame) -> pd.DataFrame:
    drug_col = "mapped_name" if "mapped_name" in response_df.columns else "drug_name"
    drugs = sorted(set(str(d).strip() for d in drug_smiles_df.index.astype(str)))
    if "DRUG_NAME" in drug_smiles_df.columns:
        drugs = sorted(set(drugs) | set(str(d).strip() for d in drug_smiles_df["DRUG_NAME"].dropna()))
    train_drugs = set(response_df[drug_col].astype(str).str.strip())
    rows = []
    for idx, drug in enumerate(sorted(set(drugs) | train_drugs)):
        rows.append(
            {
                "drug_index": idx,
                "drug_id": drug,
                "in_smiles_table": drug in set(drug_smiles_df.index.astype(str))
                or (
                    "DRUG_NAME" in drug_smiles_df.columns
                    and drug in set(drug_smiles_df["DRUG_NAME"].astype(str))
                ),
                "in_gdsc_response": drug in train_drugs,
            }
        )
    return pd.DataFrame(rows)


def build_data_alignment_report(
    response_df: pd.DataFrame,
    expression_latent_dict: dict,
    tcga_latent_dict: Optional[dict],
    eval_targets: Sequence[Tuple[str, str]] = DEFAULT_TCGA_EVAL_TARGETS,
) -> pd.DataFrame:
    rows: List[dict] = []
    model_ids = set(str(k) for k in expression_latent_dict.keys())
    gdsc_samples = set(response_df["ModelID"].astype(str)) if "ModelID" in response_df.columns else set()
    rows.append(
        {
            "domain": "CCLE",
            "dataset": "GDSC2_response",
            "n_rows": len(response_df),
            "n_unique_samples": len(gdsc_samples),
            "n_samples_with_latent": len(gdsc_samples & model_ids),
            "n_unique_drugs": response_df.get("mapped_name", response_df.get("drug_name", pd.Series())).nunique(),
        }
    )
    tcga_keys = set(tcga_latent_dict.keys()) if tcga_latent_dict else set()
    patient_from_latent = set()
    for key in tcga_keys:
        parts = str(key).split("-")
        if len(parts) >= 3:
            patient_from_latent.add("-".join(parts[:3]))

    for eval_key, path in eval_targets:
        tcga_df = load_tcga_response_csv(path)
        patients = set(tcga_df["Patient_id"].astype(str))
        rows.append(
            {
                "domain": "TCGA",
                "dataset": eval_key,
                "path": path,
                "n_rows": len(tcga_df),
                "n_unique_samples": tcga_df["Patient_id"].nunique(),
                "n_samples_with_latent": len(patients & patient_from_latent),
                "n_unique_drugs": tcga_df["drug_name"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def _per_drug_metrics_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    rows = []
    for drug_id, grp in pred_df.groupby("drug_id"):
        y_true = grp["ground_truth"].astype(float).values
        y_score = grp["confidence"].astype(float).values
        if len(np.unique(y_true)) < 2:
            auc = np.nan
            auprc = np.nan
        else:
            try:
                auc = roc_auc_score(y_true, y_score)
                auprc = average_precision_score(y_true, y_score)
            except ValueError:
                auc = np.nan
                auprc = np.nan
        rows.append(
            {
                "drug_id": drug_id,
                "n_samples": len(grp),
                "AUC": auc,
                "AUPRC": auprc,
            }
        )
    return pd.DataFrame(rows)


def target_metrics_per_drug_df(tcga_result: dict) -> pd.DataFrame:
    rows = []
    for drug, metrics in tcga_result.get("Drug_Metrics", {}).items():
        rows.append(
            {
                "drug_id": drug,
                "AUC": metrics.get("AUC", np.nan),
                "AUPRC": metrics.get("AUPRC", np.nan),
            }
        )
    return pd.DataFrame(rows)


def target_metrics_summary_df(eval_key: str, tcga_result: dict) -> pd.DataFrame:
    global_m = tcga_result.get("Global_Metrics", {})
    avg_m = tcga_result.get("Average_Metrics", {})
    drug_m = tcga_result.get("Drug_Metrics", {})
    valid_auc = [v.get("AUC") for v in drug_m.values() if v.get("AUC") is not None and not pd.isna(v.get("AUC"))]
    return pd.DataFrame(
        [
            {
                "eval_target": eval_key,
                "global_auc": global_m.get("AUC", np.nan),
                "global_auprc": global_m.get("AUPRC", np.nan),
                "global_f1": global_m.get("f1_score", np.nan),
                "average_auc": avg_m.get("AUC", np.nan),
                "average_auprc": avg_m.get("AUPRC", np.nan),
                "n_drugs_total": len(drug_m),
                "n_drugs_with_valid_auc": len(valid_auc),
            }
        ]
    )


def source_metrics_summary_df(test_metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "domain": "CCLE",
                "split": "test",
                "AUC": test_metrics.get("AUC", np.nan),
                "AUPRC": test_metrics.get("AUPRC", np.nan),
                "Loss": test_metrics.get("Loss", np.nan),
            }
        ]
    )


INTEGRATED_METRIC_KEYS = (
    "Integrated_Global_TCGA_AUC",
    "Integrated_Global_TCGA_AUPRC",
    "Integrated_Average_TCGA_AUC",
    "Integrated_Average_TCGA_AUPRC",
    "Integrated_DrugMacro_TCGA_AUC",
    "Integrated_DrugMacro_TCGA_AUPRC",
    "Integrated_n_tcga_samples_pooled",
    "Integrated_n_tcga_drugs_with_valid_auc",
    "Integrated_n_tcga_eval_targets",
)

INTEGRATED_SUMMARY_PRIORITY_COLUMNS = [
    "Model_ID",
    "Test_AUC",
    "Val_AUC",
    "Global_TCGA_AUC",
    "Average_TCGA_AUC",
    "tcga_only3_Global_TCGA_AUC",
    "tcga_only3_Average_TCGA_AUC",
    "dapl_Global_TCGA_AUC",
    "dapl_Average_TCGA_AUC",
    "TCGA2_Global_TCGA_AUC",
    "TCGA2_Average_TCGA_AUC",
    *INTEGRATED_METRIC_KEYS,
]


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def normalize_tcga_eval_result(result: Optional[dict]) -> dict:
    """Convert legacy per-drug dicts to Global/Average/Drug_Metrics structure."""
    if not result:
        return {}
    raw = result.get("_raw_inference") or result
    if raw.get("Global_Metrics") and raw.get("Drug_Metrics") is not None:
        return raw

    skip = {"Sample_Predictions", "_raw_inference", "Global_Metrics", "Average_Metrics", "Drug_Metrics"}
    drug_metrics: Dict[str, dict] = dict(raw.get("Drug_Metrics") or {})
    if not drug_metrics:
        for key, val in raw.items():
            if key in skip or not isinstance(val, dict) or "AUC" not in val:
                continue
            drug_metrics[key] = {
                "AUC": _safe_float(val.get("AUC")),
                "AUPRC": _safe_float(val.get("AUPRC")),
            }

    pooled_preds: List[float] = []
    pooled_targets: List[float] = []
    for row in raw.get("Sample_Predictions") or []:
        if not isinstance(row, dict):
            continue
        pooled_preds.append(_safe_float(row.get("confidence")))
        pooled_targets.append(_safe_float(row.get("ground_truth")))

    global_metrics: dict = {"AUC": np.nan, "AUPRC": np.nan}
    if pooled_targets and len(set(pooled_targets)) > 1:
        try:
            global_metrics["AUC"] = roc_auc_score(pooled_targets, pooled_preds)
            global_metrics["AUPRC"] = average_precision_score(pooled_targets, pooled_preds)
        except ValueError:
            pass

    aucs = [_safe_float(m.get("AUC")) for m in drug_metrics.values()]
    auprcs = [_safe_float(m.get("AUPRC")) for m in drug_metrics.values()]
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    valid_auprcs = [a for a in auprcs if not np.isnan(a)]

    return {
        "Global_Metrics": global_metrics,
        "Average_Metrics": {
            "AUC": float(np.nanmean(valid_aucs)) if valid_aucs else np.nan,
            "AUPRC": float(np.nanmean(valid_auprcs)) if valid_auprcs else np.nan,
        },
        "Drug_Metrics": drug_metrics,
        "Sample_Predictions": raw.get("Sample_Predictions") or [],
    }


def normalize_tcga_eval_suite(tcga_eval_results: Dict[str, dict]) -> Dict[str, dict]:
    return {k: normalize_tcga_eval_result(v) for k, v in (tcga_eval_results or {}).items()}


def merge_all_target_predictions(tcga_eval_results: Dict[str, dict]) -> pd.DataFrame:
    """Concatenate per-sample predictions from all TCGA eval targets."""
    from tools.prediction_export import predictions_from_tcga_inference_result

    parts = []
    for eval_key, result in tcga_eval_results.items():
        if not result:
            continue
        tag = EVAL_KEY_TO_LEGACY_TAG.get(eval_key, eval_key)
        part = predictions_from_tcga_inference_result(result, tcga_source=tag)
        if part is None or part.empty:
            continue
        part = part.copy()
        part["eval_target"] = eval_key
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def compute_integrated_tcga_metrics(tcga_eval_results: Dict[str, dict]) -> dict:
    """
    Cross-target integrated metrics:
    - Integrated_Global_*: pooled over all samples from gdsc13 + tcga_only3 + dapl
    - Integrated_Average_*: macro mean of per-target Average_Metrics
    - Integrated_DrugMacro_*: macro mean of all per-drug AUC/AUPRC with valid values
    """
    tcga_eval_results = normalize_tcga_eval_suite(tcga_eval_results)
    pooled_preds: List[float] = []
    pooled_targets: List[float] = []
    target_avg_aucs: List[float] = []
    target_avg_auprcs: List[float] = []
    drug_aucs: List[float] = []
    drug_auprcs: List[float] = []
    n_targets = 0

    merged_preds = merge_all_target_predictions(tcga_eval_results)
    if not merged_preds.empty:
        pooled_preds = merged_preds["confidence"].astype(float).tolist()
        pooled_targets = merged_preds["ground_truth"].astype(float).tolist()

    for eval_key, result in tcga_eval_results.items():
        if not result:
            continue
        n_targets += 1
        avg_m = result.get("Average_Metrics", {})
        avg_auc = _safe_float(avg_m.get("AUC"))
        avg_auprc = _safe_float(avg_m.get("AUPRC"))
        if not np.isnan(avg_auc):
            target_avg_aucs.append(avg_auc)
        if not np.isnan(avg_auprc):
            target_avg_auprcs.append(avg_auprc)
        for metrics in result.get("Drug_Metrics", {}).values():
            auc = _safe_float(metrics.get("AUC"))
            auprc = _safe_float(metrics.get("AUPRC"))
            if not np.isnan(auc):
                drug_aucs.append(auc)
            if not np.isnan(auprc):
                drug_auprcs.append(auprc)

    integrated: dict = {
        "Integrated_n_tcga_eval_targets": n_targets,
        "Integrated_n_tcga_samples_pooled": len(pooled_targets),
        "Integrated_n_tcga_drugs_with_valid_auc": len(drug_aucs),
    }

    if pooled_targets and len(set(pooled_targets)) > 1:
        try:
            integrated["Integrated_Global_TCGA_AUC"] = roc_auc_score(pooled_targets, pooled_preds)
            integrated["Integrated_Global_TCGA_AUPRC"] = average_precision_score(pooled_targets, pooled_preds)
        except ValueError:
            integrated["Integrated_Global_TCGA_AUC"] = np.nan
            integrated["Integrated_Global_TCGA_AUPRC"] = np.nan
    else:
        integrated["Integrated_Global_TCGA_AUC"] = np.nan
        integrated["Integrated_Global_TCGA_AUPRC"] = np.nan

    integrated["Integrated_Average_TCGA_AUC"] = float(np.nanmean(target_avg_aucs)) if target_avg_aucs else np.nan
    integrated["Integrated_Average_TCGA_AUPRC"] = float(np.nanmean(target_avg_auprcs)) if target_avg_auprcs else np.nan
    integrated["Integrated_DrugMacro_TCGA_AUC"] = float(np.nanmean(drug_aucs)) if drug_aucs else np.nan
    integrated["Integrated_DrugMacro_TCGA_AUPRC"] = float(np.nanmean(drug_auprcs)) if drug_auprcs else np.nan
    return integrated


def build_integrated_target_summary_df(tcga_eval_results: Dict[str, dict]) -> pd.DataFrame:
    """One row per eval target plus a final INTEGRATED_ALL row."""
    rows = []
    for eval_key, result in tcga_eval_results.items():
        if not result:
            continue
        row = target_metrics_summary_df(eval_key, result).iloc[0].to_dict()
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    integrated = compute_integrated_tcga_metrics(tcga_eval_results)
    rows.append(
        {
            "eval_target": "INTEGRATED_ALL",
            "global_auc": integrated.get("Integrated_Global_TCGA_AUC", np.nan),
            "global_auprc": integrated.get("Integrated_Global_TCGA_AUPRC", np.nan),
            "global_f1": np.nan,
            "average_auc": integrated.get("Integrated_Average_TCGA_AUC", np.nan),
            "average_auprc": integrated.get("Integrated_Average_TCGA_AUPRC", np.nan),
            "n_drugs_total": integrated.get("Integrated_n_tcga_drugs_with_valid_auc", 0),
            "n_drugs_with_valid_auc": integrated.get("Integrated_n_tcga_drugs_with_valid_auc", 0),
        }
    )
    return pd.DataFrame(rows)


def flatten_tcga_eval_metrics(tcga_eval_results: Dict[str, dict], prefix_map: Optional[dict] = None) -> dict:
    """Flatten multi-target TCGA results + integrated cross-target metrics for comparison CSV."""
    tcga_eval_results = normalize_tcga_eval_suite(tcga_eval_results)
    prefix_map = prefix_map or {
        "gdsc_intersect13": "",
        "tcga_only3": "tcga_only3_",
        "dapl": "dapl_",
    }
    flat: dict = {}
    for eval_key, result in tcga_eval_results.items():
        prefix = prefix_map.get(eval_key, f"{eval_key}_")
        if not result:
            continue
        flat[f"{prefix}Global_TCGA_AUC"] = result.get("Global_Metrics", {}).get("AUC", np.nan)
        flat[f"{prefix}Global_TCGA_AUPRC"] = result.get("Global_Metrics", {}).get("AUPRC", np.nan)
        flat[f"{prefix}Average_TCGA_AUC"] = result.get("Average_Metrics", {}).get("AUC", np.nan)
        flat[f"{prefix}Average_TCGA_AUPRC"] = result.get("Average_Metrics", {}).get("AUPRC", np.nan)
        for drug, metrics in result.get("Drug_Metrics", {}).items():
            flat[f"{prefix}{drug}_TCGA_AUC"] = metrics.get("AUC", np.nan)
            flat[f"{prefix}{drug}_TCGA_AUPRC"] = metrics.get("AUPRC", np.nan)
        # Backward-compatible TCGA2_* aliases for dapl
        if eval_key == "dapl":
            flat["TCGA2_Global_TCGA_AUC"] = flat.get("dapl_Global_TCGA_AUC", np.nan)
            flat["TCGA2_Global_TCGA_AUPRC"] = flat.get("dapl_Global_TCGA_AUPRC", np.nan)
            flat["TCGA2_Average_TCGA_AUC"] = flat.get("dapl_Average_TCGA_AUC", np.nan)
            flat["TCGA2_Average_TCGA_AUPRC"] = flat.get("dapl_Average_TCGA_AUPRC", np.nan)

    flat.update(compute_integrated_tcga_metrics(tcga_eval_results))

    # Primary aliases: gdsc_intersect13 remains the headline Global/Average columns
    gdsc = tcga_eval_results.get("gdsc_intersect13") or {}
    if gdsc:
        flat.setdefault("Global_TCGA_AUC", gdsc.get("Global_Metrics", {}).get("AUC", np.nan))
        flat.setdefault("Global_TCGA_AUPRC", gdsc.get("Global_Metrics", {}).get("AUPRC", np.nan))
        flat.setdefault("Average_TCGA_AUC", gdsc.get("Average_Metrics", {}).get("AUC", np.nan))
        flat.setdefault("Average_TCGA_AUPRC", gdsc.get("Average_Metrics", {}).get("AUPRC", np.nan))
    return flat


def is_per_drug_tcga_column(col: str) -> bool:
    """True for per-drug AUC/AUPRC columns (exclude global/average/integrated summaries)."""
    if not (col.endswith("_TCGA_AUC") or col.endswith("_TCGA_AUPRC")):
        return False
    summary_prefixes = (
        "Global_",
        "Average_",
        "Integrated_",
        "tcga_only3_Global_",
        "tcga_only3_Average_",
        "dapl_Global_",
        "dapl_Average_",
        "TCGA2_Global_",
        "TCGA2_Average_",
    )
    return not any(col.startswith(p) for p in summary_prefixes)


def write_eval_metrics_integrated_summary(outfolder: str, comparison_df: pd.DataFrame) -> Optional[str]:
    """
    Write a slim integrated-eval table for quick cross-model comparison.
    Reads from parameter_comparison_detailed or any wide comparison DataFrame.
    """
    if comparison_df is None or comparison_df.empty:
        return None
    cols = [c for c in INTEGRATED_SUMMARY_PRIORITY_COLUMNS if c in comparison_df.columns]
    extra = [
        c
        for c in comparison_df.columns
        if c.startswith("Integrated_")
        or c.startswith("tcga_only3_")
        or c.startswith("dapl_")
        or c.startswith("TCGA2_")
        or c in ("Global_TCGA_AUC", "Average_TCGA_AUC", "Test_AUC", "Val_AUC", "Model_ID", "ID")
    ]
    ordered = []
    for c in INTEGRATED_SUMMARY_PRIORITY_COLUMNS + sorted(set(extra)):
        if c in comparison_df.columns and c not in ordered:
            ordered.append(c)
    slim = comparison_df[ordered].copy()
    out_path = os.path.join(outfolder, "eval_metrics_integrated_summary.csv")
    slim.to_csv(out_path, index=False)
    return out_path


def run_tcga_eval_suite_latent(
    inference_fn,
    model_components,
    best_model_path,
    ft_params,
    tcga_latent_dict,
    drug_latent_dict,
    gin_type,
    fold_model_folder,
    drug_smiles_df,
    eval_targets: Sequence[Tuple[str, str]] = DEFAULT_TCGA_EVAL_TARGETS,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for eval_key, path in eval_targets:
        tag = EVAL_KEY_TO_LEGACY_TAG.get(eval_key, eval_key)
        results[eval_key] = inference_fn(
            model_components,
            path,
            best_model_path,
            ft_params,
            tcga_latent_dict,
            drug_latent_dict,
            gin_type,
            fold_model_folder=fold_model_folder,
            drug_smiles_df=drug_smiles_df,
            tcga_tag=tag,
        ) or {
            "Global_Metrics": {},
            "Average_Metrics": {},
            "Drug_Metrics": {},
            "Sample_Predictions": [],
        }
    return results


def predictions_from_eval_suite(tcga_eval_results: Dict[str, dict]) -> Dict[str, pd.DataFrame]:
    from tools.prediction_export import predictions_from_tcga_inference_result

    out = {}
    for eval_key, result in tcga_eval_results.items():
        tag = EVAL_KEY_TO_LEGACY_TAG.get(eval_key, eval_key)
        out[eval_key] = predictions_from_tcga_inference_result(result, tcga_source=tag)
    return out


def export_codeae_finetune_eval(
    output_dir: str,
    config: dict,
    ccle_pred_df: pd.DataFrame,
    tcga_eval_results: Dict[str, dict],
    drug_smiles_df: pd.DataFrame,
    response_df: pd.DataFrame,
    expression_latent_dict: dict,
    tcga_latent_dict: Optional[dict],
    test_metrics: Optional[dict] = None,
    metrics_history: Optional[dict] = None,
    best_model_path: Optional[str] = None,
    fold_id: Optional[int] = None,
) -> dict:
    """
    Write CODEAE-style finetune eval artifacts (no pretrain latent/tsne/kmeans).
    For 5-fold, pass fold_id (0-indexed) to write under fold_{fold_id}/.
    """
    base = output_dir if fold_id is None else os.path.join(output_dir, f"fold_{fold_id}")
    safemakedirs(base)

    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    drug_list_df = build_drug_list_df(drug_smiles_df, response_df)
    drug_list_df.to_csv(os.path.join(base, "drug_list.csv"), index=False)

    align_df = build_data_alignment_report(response_df, expression_latent_dict, tcga_latent_dict)
    align_df.to_csv(os.path.join(base, "data_alignment_report.csv"), index=False)

    if ccle_pred_df is not None and not ccle_pred_df.empty:
        ccle_pred_df.to_csv(os.path.join(base, "source_prediction_results.csv"), index=False)
        src_per_drug = _per_drug_metrics_from_predictions(ccle_pred_df)
        if not src_per_drug.empty:
            src_per_drug.to_csv(os.path.join(base, "source_metrics_per_drug.csv"), index=False)

    if test_metrics is not None:
        source_metrics_summary_df(test_metrics).to_csv(
            os.path.join(base, "source_metrics_summary.csv"), index=False
        )

    if metrics_history is not None:
        pd.DataFrame(metrics_history).to_csv(os.path.join(base, "masked_loss_log.csv"), index=False)

    if best_model_path and os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(base, "model_final.pth"))

    saved_targets = {}
    for eval_key, result in tcga_eval_results.items():
        target_dir = os.path.join(base, f"target_eval_{eval_key}")
        safemakedirs(target_dir)
        pred_df = predictions_from_eval_suite({eval_key: result}).get(eval_key, pd.DataFrame())
        if not pred_df.empty:
            pred_df.to_csv(os.path.join(target_dir, "target_prediction_results.csv"), index=False)
        per_drug = target_metrics_per_drug_df(result)
        if not per_drug.empty:
            per_drug.to_csv(os.path.join(target_dir, "target_metrics_per_drug.csv"), index=False)
        target_metrics_summary_df(eval_key, result).to_csv(
            os.path.join(target_dir, "target_metrics_summary.csv"), index=False
        )
        saved_targets[eval_key] = target_dir

    # Integrated cross-target eval (gdsc13 + tcga_only3 + dapl)
    integrated_dir = os.path.join(base, "target_eval_integrated")
    safemakedirs(integrated_dir)
    integrated_summary = build_integrated_target_summary_df(tcga_eval_results)
    if not integrated_summary.empty:
        integrated_summary.to_csv(
            os.path.join(integrated_dir, "target_metrics_summary.csv"), index=False
        )
    merged_preds = merge_all_target_predictions(tcga_eval_results)
    if not merged_preds.empty:
        merged_preds.to_csv(
            os.path.join(integrated_dir, "target_prediction_results.csv"), index=False
        )
        per_drug = _per_drug_metrics_from_predictions(merged_preds)
        if not per_drug.empty:
            per_drug.to_csv(
                os.path.join(integrated_dir, "target_metrics_per_drug.csv"), index=False
            )
    integrated_flat = compute_integrated_tcga_metrics(tcga_eval_results)
    pd.DataFrame([integrated_flat]).to_csv(
        os.path.join(base, "eval_metrics_integrated_summary.csv"), index=False
    )

    return {"base": base, "target_dirs": saved_targets, "integrated_dir": integrated_dir}


def write_target_eval_fold_mean_std(model_folder: str, n_folds: int = 5) -> Optional[str]:
    """Aggregate target_metrics_summary.csv across folds (5-fold pipelines)."""
    rows = []
    for fold in range(n_folds):
        for eval_key, _ in DEFAULT_TCGA_EVAL_TARGETS:
            summary_path = os.path.join(
                model_folder,
                f"fold_{fold}",
                f"target_eval_{eval_key}",
                "target_metrics_summary.csv",
            )
            if not os.path.exists(summary_path):
                continue
            part = pd.read_csv(summary_path)
            part["fold"] = fold
            rows.append(part)
    if not rows:
        return None
    all_df = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(model_folder, "target_eval_metrics_summary_fold_mean_std.csv")
    grouped = (
        all_df.groupby("eval_target", dropna=False)[
            ["global_auc", "global_auprc", "average_auc", "average_auprc"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(str(x) for x in col if x).strip("_") for col in grouped.columns.values]
    grouped.to_csv(out_path, index=False)
    eval_path = os.path.join(model_folder, "eval_metrics_summary_fold_mean_std.csv")
    all_df.groupby("eval_target").mean(numeric_only=True).reset_index().to_csv(eval_path, index=False)
    return out_path
