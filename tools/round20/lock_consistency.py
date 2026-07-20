"""Independent recalculation of Stage 20A/20B decisions vs stored locks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tools.round20.result_contracts import DEFAULT_RUN_ROOT, load_json, load_manifest


def _job_metrics_rows(manifest_path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for job in load_manifest(manifest_path):
        if job.get("skip_train"):
            continue
        mp = Path(job["output_dir"]) / "metrics.json"
        if not mp.is_file():
            continue
        m = load_json(mp)
        if m.get("status") != "COMPLETE" or m.get("smoke"):
            continue
        rows.append(
            {
                "job_id": job["job_id"],
                "candidate_id": job.get("candidate_id"),
                "context_id": job.get("context_id"),
                "split_seed": job["split_seed"],
                "fold": job["fold"],
                "DrugMacro_AUC": m["metrics"].get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": m["metrics"].get("DrugMacro_AUPRC"),
            }
        )
    return pd.DataFrame(rows)


def recalculate_stage20a_decision(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    parsimony_delta: float = 0.005,
    auprc_max_drop: float = 0.01,
    major_fail_auc_delta: float = -0.02,
    min_nonworse_seeds: int = 2,
) -> dict:
    stage = run_root / "stage20a_dimension"
    stored = load_json(stage / "stage20a_dimension_decision.json")
    df = _job_metrics_rows(stage / "manifest.jsonl")
    if df.empty:
        raise RuntimeError("No Stage 20A metrics for recalculation")

    pivot = df.pivot_table(
        index=["split_seed", "fold"],
        columns="context_id",
        values=["DrugMacro_AUC", "DrugMacro_AUPRC"],
        aggfunc="first",
    )
    pair_rows = []
    for (seed, fold), r in pivot.iterrows():
        pair_rows.append(
            {
                "split_seed": int(seed),
                "fold": int(fold),
                "c16_auc": r[("DrugMacro_AUC", "C16")],
                "c32_auc": r[("DrugMacro_AUC", "C32")],
                "c16_auprc": r[("DrugMacro_AUPRC", "C16")],
                "c32_auprc": r[("DrugMacro_AUPRC", "C32")],
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    mean_c16 = float(df.loc[df.context_id == "C16", "DrugMacro_AUC"].mean())
    mean_c32 = float(df.loc[df.context_id == "C32", "DrugMacro_AUC"].mean())
    mean_delta = mean_c32 - mean_c16
    mean_ap_c16 = float(df.loc[df.context_id == "C16", "DrugMacro_AUPRC"].mean())
    mean_ap_c32 = float(df.loc[df.context_id == "C32", "DrugMacro_AUPRC"].mean())
    seed_deltas = {}
    for seed, g in pairwise.groupby("split_seed"):
        seed_deltas[int(seed)] = float(g["c32_auc"].mean() - g["c16_auc"].mean())
    nonworse = sum(1 for d in seed_deltas.values() if d >= 0)
    worst_seed_delta = min(seed_deltas.values()) if seed_deltas else 0.0

    guardrails = {
        "mean_auc_nonworse": mean_delta >= 0,
        "seed_majority": nonworse >= min_nonworse_seeds,
        "auprc": (mean_ap_c32 >= mean_ap_c16 - auprc_max_drop),
        "no_major_fail": worst_seed_delta >= major_fail_auc_delta,
        "parsimony": abs(mean_delta) >= parsimony_delta,
    }

    if mean_c32 < mean_c16:
        selected, reason = "C16", "c32_mean_auc_lower"
    elif abs(mean_delta) < parsimony_delta:
        selected, reason = "C16", "parsimony"
    elif mean_delta >= parsimony_delta and guardrails["seed_majority"] and guardrails["auprc"] and guardrails["no_major_fail"]:
        selected, reason = "C32", "stable_improvement"
    else:
        selected, reason = "C16", "guardrail_not_met"

    recalc = {
        "selected_context": selected,
        "context_dim": 16 if selected == "C16" else 32,
        "omics_dim": 80 if selected == "C16" else 96,
        "mean_auc": {"C16": mean_c16, "C32": mean_c32},
        "mean_delta_c32_minus_c16": mean_delta,
        "seed_deltas": {str(k): v for k, v in seed_deltas.items()},
        "guardrails": guardrails,
        "reason": reason,
    }
    stored_ctx = stored.get("selected_context")
    matches = (
        stored_ctx == selected
        and stored.get("status") == "LOCKED"
        and abs(float(stored.get("mean_auc_delta_c32_minus_c16", 0)) - mean_delta) < 1e-9
    )
    return {
        **recalc,
        "stored_decision_matches_recalculation": matches,
        "stored_selected_context": stored_ctx,
        "stored_reason": stored.get("reason"),
    }


def recalculate_stage20b_guardrails(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    auprc_max_drop: float = 0.01,
    major_fail_auc_delta: float = -0.02,
    min_nonworse_seeds: int = 2,
) -> dict:
    stage = run_root / "stage20b_predictor"
    stored = load_json(stage / "stage20b_guardrail_report.json")
    rows = []
    for job in load_manifest(stage / "manifest.jsonl"):
        mp = Path(job["output_dir"]) / "metrics.json"
        if not mp.is_file():
            continue
        m = load_json(mp)
        if m.get("status") != "COMPLETE":
            continue
        rows.append(
            {
                "candidate_id": job["candidate_id"],
                "split_seed": job["split_seed"],
                "fold": job["fold"],
                "DrugMacro_AUC": m["metrics"].get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": m["metrics"].get("DrugMacro_AUPRC"),
            }
        )
    df = pd.DataFrame(rows)
    e3 = df[df.candidate_id == "B_E3"]
    gated = df[df.candidate_id == "B_GATED"]
    mean_auc_delta = float(gated["DrugMacro_AUC"].mean() - e3["DrugMacro_AUC"].mean())
    mean_ap_delta = float(gated["DrugMacro_AUPRC"].mean() - e3["DrugMacro_AUPRC"].mean())
    seed_deltas = {}
    for seed in sorted(df["split_seed"].unique()):
        g = gated[gated.split_seed == seed]["DrugMacro_AUC"].mean()
        b = e3[e3.split_seed == seed]["DrugMacro_AUC"].mean()
        seed_deltas[str(int(seed))] = float(g - b)
    nonworse = sum(1 for d in seed_deltas.values() if d >= 0)
    worst = min(seed_deltas.values()) if seed_deltas else 0.0
    guardrails = {
        "g1_mean_auc_nonworse": mean_auc_delta >= 0,
        "g2_seed_majority": nonworse >= min_nonworse_seeds,
        "g3_auprc": mean_ap_delta >= -auprc_max_drop,
        "g4_no_major_fail": worst >= major_fail_auc_delta,
        "g5_complete": len(gated) == 15 and len(e3) == 15,
    }
    all_pass = all(guardrails.values())
    recalc = {
        "candidate": "B_GATED",
        "baseline": "B_E3",
        "mean_auc_delta": mean_auc_delta,
        "mean_auprc_delta": mean_ap_delta,
        "seed_auc_deltas": seed_deltas,
        "guardrails": guardrails,
        "all_pass": all_pass,
    }
    matches = (
        bool(stored.get("all_pass")) == all_pass
        and abs(float(stored.get("mean_auc_delta", 0)) - mean_auc_delta) < 1e-9
    )
    return {**recalc, "stored_guardrails_match_recalculation": matches}


def verify_final_lock_consistency(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    lock_path = run_root / "stage20c_lock/final_model_lock.json"
    lock = load_json(lock_path)
    a = recalculate_stage20a_decision(run_root=run_root)
    b = recalculate_stage20b_guardrails(run_root=run_root)
    from tools.round20_model_lock import select_final_model as _sel

    expected_model, reason = _sel(b, parsimony_threshold=0.005)

    forbidden = []
    ctx_ok = lock["selected_context"]["id"] == a["selected_context"]
    model_ok = lock["selected_model"]["candidate_id"] == expected_model
    reason_ok = lock.get("selection_reason") == reason or (
        lock.get("selection_reason") == "gated_failed_guardrails" and reason == "gated_failed_guardrails"
    )
    if lock.get("forbidden_metrics_used"):
        forbidden.append("forbidden_metrics_used_true")
    return {
        "context_matches_stage20a": ctx_ok,
        "model_matches_stage20b": model_ok,
        "reason_matches": reason_ok,
        "expected_model": expected_model,
        "expected_reason": reason,
        "forbidden_flags": forbidden,
        "ok": ctx_ok and model_ok and reason_ok and not forbidden,
    }
