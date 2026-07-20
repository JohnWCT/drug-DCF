"""TCGA provenance, ensemble recalculation, and metric validation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable
from tools.round20.result_contracts import DEFAULT_RUN_ROOT, load_json, sha256_file


TCGA_TARGET_SUFFIXES = (
    "gdsc_intersect13",
    "tcga_only3",
    "dapl",
    "aacdr_tcga_only",
    "aacdr_gdsc_intersect",
)


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit_tcga_provenance(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    lock_path = run_root / "stage20c_lock/final_model_lock.json"
    tcga_dir = run_root / "stage20d_tcga"
    lock = load_json(lock_path)
    lock_sha = sha256_file(lock_path)
    lock_ts = _parse_ts(lock.get("created_at"))

    # Best-effort inference start from earliest job log under stage20d or dispatch
    inference_started = None
    for p in sorted(tcga_dir.glob("*.csv")):
        ts = datetime.utcfromtimestamp(p.stat().st_mtime)
        inference_started = ts if inference_started is None else min(inference_started, ts)

    timing_ok = True
    if lock_ts and inference_started:
        timing_ok = lock_ts < inference_started.replace(tzinfo=lock_ts.tzinfo)

    from tools.round20.checkpoint_inventory import resolve_locked_checkpoints
    ckpts = resolve_locked_checkpoints(run_root=run_root)
    preflight = load_json(tcga_dir / "stage20d_tcga_preflight.json") if (
        tcga_dir / "stage20d_tcga_preflight.json"
    ).is_file() else {}

    return {
        "model_lock_sha256": lock_sha,
        "model_lock_created_at": lock.get("created_at"),
        "inference_started_at_estimate": inference_started.isoformat() if inference_started else None,
        "lock_before_inference": timing_ok,
        "selected_context": lock["selected_context"]["id"],
        "selected_predictor": lock["selected_model"]["candidate_id"],
        "checkpoint_sha256": [c["sha256"] for c in ckpts if c["sha256"]],
        "preflight_status": preflight.get("status"),
        "status": "PASS" if timing_ok else "FAIL",
    }


def _ensemble_from_checkpoints(df_ckpt: pd.DataFrame) -> pd.Series:
    prob_cols = [c for c in df_ckpt.columns if c.startswith("prob_ckpt")]
    return df_ckpt[prob_cols].mean(axis=1)


def _row_id_series(df: pd.DataFrame) -> pd.Series:
    if "_row_id" in df.columns:
        return df["_row_id"]
    if "row_id" in df.columns:
        return df["row_id"]
    raise KeyError("prediction CSV missing row_id/_row_id column")


def audit_tcga_predictions(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    rtol: float = 1e-7,
    atol: float = 1e-8,
) -> dict:
    tcga_dir = run_root / "stage20d_tcga"
    target_reports = []
    all_ok = True
    for suffix in TCGA_TARGET_SUFFIXES:
        ens_path = tcga_dir / f"predictions_ensemble__{suffix}.csv"
        ckpt_path = tcga_dir / f"predictions_by_checkpoint__{suffix}.csv"
        if not ens_path.is_file() or not ckpt_path.is_file():
            target_reports.append({"target": suffix, "status": "MISSING", "ok": False})
            all_ok = False
            continue
        ens = pd.read_csv(ens_path)
        ckpt = pd.read_csv(ckpt_path)
        prob_cols = [c for c in ckpt.columns if c.startswith("prob_ckpt")]
        row_col = "_row_id" if "_row_id" in ckpt.columns else "row_id"
        recomputed = _ensemble_from_checkpoints(ckpt.set_index(row_col))
        ens_idx = _row_id_series(ens)
        merged = ens.set_index(ens_idx).join(recomputed.rename("recomputed"), how="inner")
        diff = (merged["prediction_probability"] - merged["recomputed"]).abs()
        max_diff = float(diff.max()) if len(diff) else 0.0
        ensemble_ok = bool(np.allclose(
            merged["prediction_probability"].to_numpy(),
            merged["recomputed"].to_numpy(),
            rtol=rtol,
            atol=atol,
        ))
        nan_probs = int(ens["prediction_probability"].isna().sum())
        oob = int(((ens["prediction_probability"] < 0) | (ens["prediction_probability"] > 1)).sum())
        dup = int(ens_idx.duplicated().sum())
        ck_count_ok = bool((ens["checkpoint_count"] == len(prob_cols)).all()) if "checkpoint_count" in ens.columns else True
        row_ok = nan_probs == 0 and oob == 0 and dup == 0 and ensemble_ok and ck_count_ok
        if not row_ok:
            all_ok = False
        target_reports.append(
            {
                "target": suffix,
                "n_rows": int(len(ens)),
                "n_checkpoints": len(prob_cols),
                "max_ensemble_abs_diff": max_diff,
                "nan_probs": nan_probs,
                "out_of_range": oob,
                "duplicate_rows": dup,
                "ensemble_recalc_ok": ensemble_ok,
                "status": "PASS" if row_ok else "FAIL",
                "ok": row_ok,
            }
        )
    return {"targets": target_reports, "status": "PASS" if all_ok else "FAIL"}


def recalculate_tcga_metrics(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    tcga_dir = run_root / "stage20d_tcga"
    stored_path = tcga_dir / "tcga_metrics.json"
    stored = load_json(stored_path) if stored_path.is_file() else {}
    recalculated: Dict[str, dict] = {}
    mismatches = []
    for suffix in TCGA_TARGET_SUFFIXES:
        ens_path = tcga_dir / f"predictions_ensemble__{suffix}.csv"
        if not ens_path.is_file():
            continue
        ens = pd.read_csv(ens_path)
        rename_map = {
            "drug_id": "DRUG_NAME",
            "drug_name": "DRUG_NAME",
            "true_label": "Label",
            "prediction_probability": "probability",
        }
        metric_df = ens.rename(columns=rename_map)
        metrics = metrics_to_jsonable(calculate_robust_drug_macro_metrics(metric_df))
        recalculated[suffix] = metrics
        if suffix in stored:
            for key in ("DrugMacro_AUC", "DrugMacro_AUPRC", "Global_AUC", "Global_AUPRC"):
                s_val = stored[suffix].get(key)
                r_val = metrics.get(key)
                if s_val is None and r_val is None:
                    continue
                if s_val is None or r_val is None or abs(float(s_val) - float(r_val)) > 1e-6:
                    mismatches.append(f"{suffix}.{key}: stored={s_val} recalc={r_val}")
    return {
        "recalculated": recalculated,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def export_aggregate_artifacts(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    """Write aggregate_metrics.json and per_drug_metrics.csv from tcga_metrics."""
    tcga_dir = run_root / "stage20d_tcga"
    metrics = load_json(tcga_dir / "tcga_metrics.json")
    agg_path = tcga_dir / "aggregate_metrics.json"
    agg_path.write_text(
        __import__("json").dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    per_drug_rows = []
    for target, payload in metrics.items():
        for rec in payload.get("per_drug_records", []):
            per_drug_rows.append({"target_key": target, **rec})
    per_drug_path = tcga_dir / "per_drug_metrics.csv"
    pd.DataFrame(per_drug_rows).to_csv(per_drug_path, index=False)
    return {"aggregate_metrics": str(agg_path), "per_drug_metrics": str(per_drug_path), "n_per_drug_rows": len(per_drug_rows)}
