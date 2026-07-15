#!/usr/bin/env python3
"""Analyze Round 19E shift-validation results (per-shift, no pooled significance)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tools.round19_manifest_validator import assert_selection_frame_has_no_tcga

STRATEGIES = ("drug_heldout", "scaffold_heldout", "cancer_type_heldout")

PAIRED = [
    ("E1", "E0", "E1_minus_E0"),
    ("E2", "E1", "E2_minus_E1"),
    ("E1", "E3", "E1_minus_E3"),
    ("E4", "E1", "E4_minus_E1"),
    ("E4", "E2", "E4_minus_E2"),
    ("E5", "E0", "E5_minus_E0"),
    ("E5", "E3", "E5_minus_E3"),
]


def _load_strategy_metrics(root: Path, strategy: str) -> pd.DataFrame:
    man = root / "manifests" / f"stage19e_{strategy}_manifest.csv"
    if not man.is_file():
        raise FileNotFoundError(man)
    man_df = pd.read_csv(man)
    rows = []
    for _, job in man_df.iterrows():
        rd = Path(str(job["result_dir"]))
        st_path = rd / "job_status.json"
        metrics_path = rd / "val_metrics.json"
        summary_path = rd / "train_summary.json"
        status = "missing"
        if st_path.is_file():
            status = json.loads(st_path.read_text(encoding="utf-8")).get("status", "unknown")
        metrics = {}
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = {}
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "job_id": job["job_id"],
            "candidate_id": job["candidate_id"],
            "source_candidate_id": job.get("source_candidate_id"),
            "shift_strategy": strategy,
            "fold_id": int(job["fold_id"]),
            "drug_id": job["drug_id"],
            "predictor_id": job["predictor_id"],
            "omics_id": job["omics_id"],
            "status": status,
            "DrugMacro_AUC": metrics.get("DrugMacro_AUC"),
            "DrugMacro_AUPRC": metrics.get("DrugMacro_AUPRC"),
            "Global_AUC": metrics.get("Global_AUC"),
            "Global_AUPRC": metrics.get("Global_AUPRC"),
            "fallback_used": summary.get("fallback_used", metrics.get("fallback_used")),
            "fallback_reason": summary.get("fallback_reason", metrics.get("fallback_reason")),
            "n_valid_drugs": metrics.get("n_valid_auc_drugs", metrics.get("n_valid_drugs")),
            "Brier": metrics.get("Brier") or metrics.get("brier_score"),
            "ECE": metrics.get("ECE") or metrics.get("expected_calibration_error"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _guardrail(delta: float) -> str:
    if delta >= 0.003:
        return "PASS"
    if delta > -0.003:
        return "NON_WORSE"
    return "FAIL"


def analyze_stage19e(outdir: str, *, require_complete: bool = False) -> dict:
    root = Path(outdir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    frames = []
    for strategy in STRATEGIES:
        frames.append(_load_strategy_metrics(root, strategy))
    metrics = pd.concat(frames, ignore_index=True)
    assert_selection_frame_has_no_tcga(metrics)
    metrics.to_csv(reports / "round19e_per_fold_metrics.csv", index=False)

    n_total = int(len(metrics))
    n_done = int((metrics["status"] == "done").sum())
    if require_complete and n_done != n_total:
        raise RuntimeError(f"Stage19E incomplete: {n_done}/{n_total}")
    elif n_done < n_total:
        warnings.warn(f"Stage19E partial: {n_done}/{n_total}", stacklevel=2)

    done = metrics[metrics["status"] == "done"].copy()
    summary_rows = []
    for (cand, strategy), g in done.groupby(["candidate_id", "shift_strategy"]):
        summary_rows.append(
            {
                "candidate_id": cand,
                "shift_strategy": strategy,
                "n_folds": int(g["fold_id"].nunique()),
                "mean_DrugMacro_AUC": float(g["DrugMacro_AUC"].dropna().mean()) if g["DrugMacro_AUC"].notna().any() else None,
                "std_DrugMacro_AUC": float(g["DrugMacro_AUC"].std(ddof=0)) if len(g) > 1 else 0.0,
                "mean_DrugMacro_AUPRC": float(g["DrugMacro_AUPRC"].mean())
                if g["DrugMacro_AUPRC"].notna().any()
                else None,
                "mean_Global_AUC": float(g["Global_AUC"].mean()) if g["Global_AUC"].notna().any() else None,
                "mean_Global_AUPRC": float(g["Global_AUPRC"].mean())
                if g["Global_AUPRC"].notna().any()
                else None,
                "mean_Brier": float(g["Brier"].mean()) if g["Brier"].notna().any() else None,
                "mean_ECE": float(g["ECE"].mean()) if g["ECE"].notna().any() else None,
                "fallback_fold_count": int(g["fallback_used"].fillna(False).astype(bool).sum())
                if "fallback_used" in g
                else 0,
            }
        )
    per_shift = pd.DataFrame(summary_rows)
    per_shift.to_csv(reports / "round19e_per_shift_summary.csv", index=False)

    # Paired fold deltas — separately per shift
    pair_rows = []
    for strategy in STRATEGIES:
        sub = done[done.shift_strategy == strategy]
        for a, b, label in PAIRED:
            if a not in set(sub.candidate_id) or b not in set(sub.candidate_id):
                continue
            for fold in sorted(sub.fold_id.unique()):
                ra = sub[(sub.candidate_id == a) & (sub.fold_id == fold)]
                rb = sub[(sub.candidate_id == b) & (sub.fold_id == fold)]
                if len(ra) != 1 or len(rb) != 1:
                    continue
                if pd.isna(ra.DrugMacro_AUC.iloc[0]) or pd.isna(rb.DrugMacro_AUC.iloc[0]):
                    continue
                da = float(ra.DrugMacro_AUC.iloc[0])
                db = float(rb.DrugMacro_AUC.iloc[0])
                pair_rows.append(
                    {
                        "comparison": label,
                        "shift_strategy": strategy,
                        "fold_id": int(fold),
                        "auc_a": da,
                        "auc_b": db,
                        "delta_auc": da - db,
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(reports / "round19e_paired_fold_deltas.csv", index=False)

    # Guardrails vs E0 and E3
    guard_rows = []
    for strategy in STRATEGIES:
        base = per_shift[per_shift.shift_strategy == strategy]
        if base.empty:
            continue
        e0 = base[base.candidate_id == "E0"]
        e3 = base[base.candidate_id == "E3"]
        e0_auc = float(e0.mean_DrugMacro_AUC.iloc[0]) if len(e0) and pd.notna(e0.mean_DrugMacro_AUC.iloc[0]) else None
        e3_auc = float(e3.mean_DrugMacro_AUC.iloc[0]) if len(e3) and pd.notna(e3.mean_DrugMacro_AUC.iloc[0]) else None
        for _, row in base.iterrows():
            cid = row.candidate_id
            auc = float(row.mean_DrugMacro_AUC) if pd.notna(row.mean_DrugMacro_AUC) else None
            if auc is None:
                continue
            d0 = (auc - e0_auc) if e0_auc is not None else None
            d3 = (auc - e3_auc) if e3_auc is not None else None
            major = bool(d0 is not None and d0 <= -0.015)
            guard_rows.append(
                {
                    "candidate_id": cid,
                    "shift_strategy": strategy,
                    "mean_DrugMacro_AUC": auc,
                    "delta_vs_E0": d0,
                    "guardrail_vs_E0": _guardrail(d0) if d0 is not None else None,
                    "delta_vs_E3": d3,
                    "guardrail_vs_E3": _guardrail(d3) if d3 is not None else None,
                    "MAJOR_FAIL": major,
                }
            )
    guards = pd.DataFrame(guard_rows)
    guards.to_csv(reports / "round19e_shift_guardrails.csv", index=False)

    # Convenience slices
    for strategy, name in [
        ("drug_heldout", "round19e_drug_generalization.csv"),
        ("scaffold_heldout", "round19e_scaffold_generalization.csv"),
        ("cancer_type_heldout", "round19e_cancer_type_generalization.csv"),
    ]:
        per_shift[per_shift.shift_strategy == strategy].to_csv(reports / name, index=False)

    fallback = (
        done.groupby(["candidate_id", "shift_strategy"], as_index=False)
        .agg(
            n_folds=("fold_id", "nunique"),
            fallback_folds=("fallback_used", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        )
        if not done.empty
        else pd.DataFrame()
    )
    fallback.to_csv(reports / "round19e_fallback_summary.csv", index=False)

    calib = per_shift[["candidate_id", "shift_strategy", "mean_Brier", "mean_ECE"]].copy()
    calib.to_csv(reports / "round19e_calibration_summary.csv", index=False)

    # Resource from status CSVs if present
    res_rows = []
    for strategy in STRATEGIES:
        st = root / "manifests" / f"stage19e_{strategy}_job_status.csv"
        if st.is_file():
            sdf = pd.read_csv(st)
            res_rows.append(
                {
                    "shift_strategy": strategy,
                    "n_jobs": int(len(sdf)),
                    "n_done": int((sdf["status"] == "done").sum()) if "status" in sdf.columns else None,
                    "mean_elapsed_sec": float(sdf["elapsed_sec"].mean())
                    if "elapsed_sec" in sdf.columns
                    else None,
                }
            )
    pd.DataFrame(res_rows).to_csv(reports / "round19e_resource_summary.csv", index=False)

    summary = {
        "stage": "19e",
        "n_done": n_done,
        "n_total": n_total,
        "n_failed": int((metrics["status"] == "failed").sum()),
        "strategies": list(STRATEGIES),
        "formal_selection_lock": "NO-GO",
        "internal_test_used_for_selection": False,
        "tcga_used_for_selection": False,
        "reports_dir": str(reports),
    }
    (reports / "round19e_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
