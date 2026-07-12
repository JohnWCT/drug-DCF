"""Round 18 robust DrugMacro / Global CV metrics."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def drug_is_valid(
    n_samples: int,
    n_positive: int,
    n_negative: int,
    *,
    min_samples: int = 10,
    min_positive: int = 2,
    min_negative: int = 2,
) -> bool:
    return (
        n_samples >= min_samples
        and n_positive >= min_positive
        and n_negative >= min_negative
    )


def calculate_robust_drug_macro_metrics(
    prediction_df: pd.DataFrame,
    drug_col: str = "DRUG_NAME",
    label_col: str = "Label",
    probability_col: str = "probability",
    min_samples: int = 10,
    min_positive: int = 2,
    min_negative: int = 2,
) -> Dict[str, Any]:
    """
    Compute DrugMacro AUC/AUPRC over drugs meeting support thresholds.
    Drugs that fail thresholds are marked insufficient_class_support and excluded.
    """
    required = {drug_col, label_col, probability_col}
    missing = required - set(prediction_df.columns)
    if missing:
        raise KeyError(f"prediction_df missing columns: {sorted(missing)}")

    df = prediction_df[[drug_col, label_col, probability_col]].copy()
    df[label_col] = df[label_col].astype(int)
    df[probability_col] = df[probability_col].astype(float)

    per_drug = []
    aucs = []
    auprcs = []
    for drug, g in df.groupby(drug_col, sort=False):
        y = g[label_col].to_numpy()
        p = g[probability_col].to_numpy()
        n_samples = int(len(g))
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        valid = drug_is_valid(
            n_samples,
            n_pos,
            n_neg,
            min_samples=min_samples,
            min_positive=min_positive,
            min_negative=min_negative,
        )
        auc = _safe_auc(y, p) if valid else None
        auprc = _safe_auprc(y, p) if valid else None
        status = "ok" if valid and auc is not None else "insufficient_class_support"
        if valid and auc is not None:
            aucs.append(auc)
        if valid and auprc is not None:
            auprcs.append(auprc)
        per_drug.append(
            {
                "drug": drug,
                "n_samples": n_samples,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "valid": bool(valid),
                "status": status,
                "AUC": auc,
                "AUPRC": auprc,
            }
        )

    y_all = df[label_col].to_numpy()
    p_all = df[probability_col].to_numpy()
    global_auc = _safe_auc(y_all, p_all)
    global_auprc = _safe_auprc(y_all, p_all)

    n_total = int(df[drug_col].nunique())
    n_valid_auc = len(aucs)
    n_valid_auprc = len(auprcs)
    return {
        "DrugMacro_AUC": float(np.mean(aucs)) if aucs else None,
        "DrugMacro_AUPRC": float(np.mean(auprcs)) if auprcs else None,
        "Global_AUC": global_auc,
        "Global_AUPRC": global_auprc,
        "n_valid_auc_drugs": n_valid_auc,
        "n_valid_auprc_drugs": n_valid_auprc,
        "n_total_drugs": n_total,
        "valid_drug_fraction": float(n_valid_auc / max(n_total, 1)),
        "per_drug": pd.DataFrame(per_drug),
    }


def early_stop_score(
    metrics: Dict[str, Any],
    *,
    min_valid_drugs_for_early_stop: int = 3,
) -> Dict[str, Any]:
    """Prefer Robust DrugMacro AUC; fallback to Global AUC when support is thin."""
    n_valid = int(metrics.get("n_valid_auc_drugs") or 0)
    if n_valid >= min_valid_drugs_for_early_stop and metrics.get("DrugMacro_AUC") is not None:
        return {
            "score": float(metrics["DrugMacro_AUC"]),
            "score_name": "Robust_DrugMacro_AUC",
            "fallback_used": False,
        }
    if metrics.get("Global_AUC") is None:
        return {"score": float("nan"), "score_name": "unavailable", "fallback_used": True}
    return {
        "score": float(metrics["Global_AUC"]),
        "score_name": "Global_AUC",
        "fallback_used": True,
    }


def metrics_to_jsonable(metrics):
    """Drop DataFrame fields so metrics can be JSON-serialized."""
    out = {}
    for k, v in dict(metrics).items():
        if isinstance(v, pd.DataFrame):
            out[k + "_records"] = v.to_dict(orient="records")
            continue
        if hasattr(v, "item") and not isinstance(v, (bytes, str)):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        out[k] = v
    return out
