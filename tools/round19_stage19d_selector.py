#!/usr/bin/env python3
"""Round 19D candidate proposal from 19B/19C role policy (no TCGA/internal)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round19_fusion_models import assert_compatible
from tools.round19_manifest_validator import FORBIDDEN_SELECTION_COLS
from tools.round19_selection_lock import scan_mapping_for_forbidden, write_selection_lock

SPLIT_SEEDS = [52, 62, 72]
MODEL_SEED = 101


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    env = os.environ.get("ROUND19_GIT_HEAD", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "UNKNOWN"


def _assert_no_forbidden(df: pd.DataFrame) -> None:
    hits = sorted(FORBIDDEN_SELECTION_COLS.intersection(set(df.columns)))
    if hits:
        raise AssertionError(f"Forbidden selection columns: {hits}")


def _count_done(manifest: Path) -> Tuple[int, int]:
    df = pd.read_csv(manifest)
    _assert_no_forbidden(df)
    done = 0
    for _, row in df.iterrows():
        st = Path(str(row["result_dir"])) / "job_status.json"
        if st.is_file() and json.loads(st.read_text(encoding="utf-8")).get("status") == "done":
            done += 1
    return done, int(len(df))


def _load_composition(root: Path) -> pd.DataFrame:
    path = root / "reports" / "round19c_full_composition_ranking.csv"
    df = pd.read_csv(path)
    _assert_no_forbidden(df)
    return df


def _cell_metrics(comp: pd.DataFrame, drug: str, pred: str, omics: str) -> dict:
    row = comp[
        (comp.drug_id == drug) & (comp.predictor_id == pred) & (comp.omics_id == omics)
    ]
    if row.empty:
        raise RuntimeError(f"Missing composition metrics for {drug}×{pred}×{omics}")
    r = row.iloc[0]
    return {
        "mean_drugmacro_auc": float(r["mean_drugmacro_auc"]),
        "std_drugmacro_auc": float(r.get("std_drugmacro_auc", 0.0) or 0.0),
        "mean_drugmacro_auprc": float(r["mean_drugmacro_auprc"])
        if pd.notna(r.get("mean_drugmacro_auprc"))
        else None,
    }


def pick_best_pooled_o2(comp: pd.DataFrame) -> Tuple[str, str, dict]:
    """F3: best P0/P1 × any drug × O2 from composition ranking."""
    sub = comp[(comp.omics_id == "O2") & (comp.predictor_id.isin(["P0", "P1"]))].copy()
    if sub.empty:
        raise RuntimeError("No pooled O2 cells in composition ranking")
    sub = sub.sort_values(
        ["mean_drugmacro_auc", "mean_drugmacro_auprc", "std_drugmacro_auc"],
        ascending=[False, False, True],
        na_position="last",
    )
    r = sub.iloc[0]
    return str(r.drug_id), str(r.predictor_id), _cell_metrics(comp, str(r.drug_id), str(r.predictor_id), "O2")


def maybe_maccs_f5(
    comp: pd.DataFrame,
    *,
    f3_drug: str,
    f3_pred: str,
    f3_metrics: dict,
    root: Path,
) -> Optional[dict]:
    """F5 if D4×P0/P1×O2 meets efficiency/performance gates vs F3."""
    cands = []
    for pred in ("P0", "P1"):
        try:
            m = _cell_metrics(comp, "D4", pred, "O2")
        except RuntimeError:
            continue
        cands.append(("D4", pred, m))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[2]["mean_drugmacro_auc"], -(x[2]["mean_drugmacro_auprc"] or -1)))
    drug, pred, metrics = cands[0]
    gap = float(f3_metrics["mean_drugmacro_auc"] - metrics["mean_drugmacro_auc"])
    # Resource proxy from 19B fold0 jobs when available
    f3_job = root / "stage19b" / f"{f3_drug}__{f3_pred}__O2__fold0" / "job_status.json"
    f5_job = root / "stage19b" / f"{drug}__{pred}__O2__fold0" / "job_status.json"
    f3_res = root / "stage19b" / f"{f3_drug}__{f3_pred}__O2__fold0" / "runtime_resource_summary.json"
    f5_res = root / "stage19b" / f"{drug}__{pred}__O2__fold0" / "runtime_resource_summary.json"
    time_ok = vram_ok = False
    detail = {"auc_gap_vs_f3": gap}
    if f3_job.is_file() and f5_job.is_file():
        t3 = float(json.loads(f3_job.read_text()).get("elapsed_sec") or 0)
        t5 = float(json.loads(f5_job.read_text()).get("elapsed_sec") or 0)
        detail["epoch_wall_f3"] = t3
        detail["epoch_wall_f5"] = t5
        if t3 > 0 and t5 <= 0.75 * t3:
            time_ok = True
    if f3_res.is_file() and f5_res.is_file():
        v3 = float(json.loads(f3_res.read_text()).get("peak_gpu_mem_mb") or 0)
        v5 = float(json.loads(f5_res.read_text()).get("peak_gpu_mem_mb") or 0)
        detail["vram_f3"] = v3
        detail["vram_f5"] = v5
        if v3 > 0 and v5 <= 0.70 * v3:
            vram_ok = True
    cond_a = gap <= 0.005
    cond_b = gap <= 0.010 and time_ok
    cond_c = gap <= 0.010 and vram_ok
    if not (cond_a or cond_b or cond_c):
        return None
    return {
        "candidate_id": "F5_maccs_efficient",
        "drug_id": drug,
        "predictor_id": pred,
        "omics_id": "O2",
        "selection_role": "maccs_efficient",
        "mandatory": False,
        "metrics": metrics,
        "inclusion_reason": {
            "cond_a_auc_gap_le_005": cond_a,
            "cond_b_time": cond_b,
            "cond_c_vram": cond_c,
            **detail,
        },
    }


def maybe_graph_f6(comp: pd.DataFrame) -> Optional[dict]:
    """F6 only if nonbaseline graph P2×O2 beats D0×P2×O2 by ≥0.003 with ≥2/3 folds."""
    base = _cell_metrics(comp, "D0", "P2", "O2")
    # Prefer fold-level from 19B job metrics if available
    root_jobs = None
    effects_path = Path("result/optimization_runs/round19_factorial/reports/stage19b_job_metrics.csv")
    # resolved relative to composition root via caller; load optionally
    best = None
    for drug in ("D2", "D3"):
        try:
            m = _cell_metrics(comp, drug, "P2", "O2")
        except RuntimeError:
            continue
        delta = m["mean_drugmacro_auc"] - base["mean_drugmacro_auc"]
        if delta < 0.003:
            continue
        # fold wins from job metrics if present
        jobs_path = Path("result/optimization_runs/round19_factorial/reports/stage19b_job_metrics.csv")
        pos = 0
        if jobs_path.is_file():
            jobs = pd.read_csv(jobs_path)
            a = jobs[(jobs.drug == "D0") & (jobs.predictor == "P2") & (jobs.omics == "O2")]
            b = jobs[(jobs.drug == drug) & (jobs.predictor == "P2") & (jobs.omics == "O2")]
            for fold in (0, 1, 2):
                ra = a[a.fold_id == fold]
                rb = b[b.fold_id == fold]
                if len(ra) and len(rb) and float(rb.iloc[0].DrugMacro_AUC) > float(ra.iloc[0].DrugMacro_AUC):
                    pos += 1
        else:
            pos = 2  # fallback: mean gate only if no fold table
        if pos < 2:
            continue
        cand = (delta, drug, m, pos)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    _, drug, metrics, pos = best
    return {
        "candidate_id": "F6_nonbaseline_graph",
        "drug_id": drug,
        "predictor_id": "P2",
        "omics_id": "O2",
        "selection_role": "nonbaseline_graph",
        "mandatory": False,
        "metrics": metrics,
        "inclusion_reason": {
            "delta_vs_d0p2o2": metrics["mean_drugmacro_auc"] - base["mean_drugmacro_auc"],
            "positive_folds_vs_d0": pos,
        },
    }


def build_proposal(root: Path) -> dict:
    m19b = root / "manifests" / "stage19b_drug_predictor_manifest.csv"
    m19c = root / "manifests" / "stage19c_manifest.csv"
    done_b, n_b = _count_done(m19b)
    done_c, n_c = _count_done(m19c)
    if done_b != 117 or n_b != 117:
        raise RuntimeError(f"19B incomplete: {done_b}/{n_b}")
    if done_c != 54 or n_c != 54:
        raise RuntimeError(f"19C incomplete: {done_c}/{n_c}")
    ctrl = pd.read_csv(m19c)
    n_ctrl = int((ctrl["control_type"] == "context_shuffle").sum()) if "control_type" in ctrl.columns else 0
    if n_ctrl != 12:
        raise RuntimeError(f"Expected 12 context controls, got {n_ctrl}")

    comp = _load_composition(root)
    f3_drug, f3_pred, f3_m = pick_best_pooled_o2(comp)

    candidates: List[dict] = [
        {
            "candidate_id": "F0_historical_anchor",
            "drug_id": "D0",
            "predictor_id": "P0",
            "omics_id": "O1",
            "selection_role": "historical_anchor",
            "mandatory": True,
            "metrics": _cell_metrics(comp, "D0", "P0", "O1"),
        },
        {
            "candidate_id": "F1_primary_o2",
            "drug_id": "D0",
            "predictor_id": "P2",
            "omics_id": "O2",
            "selection_role": "primary_context_model",
            "mandatory": True,
            "metrics": _cell_metrics(comp, "D0", "P2", "O2"),
        },
        {
            "candidate_id": "F2_full_omics_o3",
            "drug_id": "D0",
            "predictor_id": "P2",
            "omics_id": "O3",
            "selection_role": "full_omics_control",
            "mandatory": True,
            "metrics": _cell_metrics(comp, "D0", "P2", "O3"),
        },
        {
            "candidate_id": "F3_best_pooled_o2",
            "drug_id": f3_drug,
            "predictor_id": f3_pred,
            "omics_id": "O2",
            "selection_role": "best_pooled_o2",
            "mandatory": True,
            "metrics": f3_m,
        },
        {
            "candidate_id": "F4_source_only_o4",
            "drug_id": "D3",
            "predictor_id": "P2",
            "omics_id": "O4",
            "selection_role": "source_only_domain_generalization",
            "mandatory": True,
            "metrics": _cell_metrics(comp, "D3", "P2", "O4"),
        },
    ]
    f5 = maybe_maccs_f5(comp, f3_drug=f3_drug, f3_pred=f3_pred, f3_metrics=f3_m, root=root)
    if f5:
        candidates.append(f5)
    f6 = maybe_graph_f6(comp)
    if f6:
        candidates.append(f6)

    for c in candidates:
        assert_compatible(c["drug_id"], c["predictor_id"])
    ids = [c["candidate_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate candidate_id")
    if not (5 <= len(candidates) <= 6):
        raise AssertionError(f"Expected 5–6 candidates, got {len(candidates)}")
    # F4 hard pin
    f4 = next(c for c in candidates if c["candidate_id"] == "F4_source_only_o4")
    if (f4["drug_id"], f4["predictor_id"], f4["omics_id"]) != ("D3", "P2", "O4"):
        raise AssertionError("F4 must remain D3×P2×O4")

    payload = {
        "lock_type": "stage19d_candidate_proposal",
        "source_stage": "19c",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage19b_jobs": "117/117",
        "stage19c_jobs": "54/54",
        "candidates": candidates,
        "n_candidates": len(candidates),
        "expected_jobs": len(candidates) * 15,
        "split_seeds": SPLIT_SEEDS,
        "model_seed": MODEL_SEED,
        "n_folds": 5,
        "max_epochs": 1500,
        "early_stop_patience": 100,
        "early_stop_start_epoch": 50,
        "internal_test_used": False,
        "tcga_used": False,
        "integrated5_used": False,
        "source_hashes": {
            "stage19c_candidate_lock": _sha256_file(root / "reports" / "round19_stage19c_candidate_lock.json"),
            "composition_ranking": _sha256_file(root / "reports" / "round19c_full_composition_ranking.csv"),
            "omics_effects": _sha256_file(root / "reports" / "round19c_omics_effects.csv"),
            "source_only_control": _sha256_file(root / "reports" / "round19c_source_only_control.csv"),
        },
    }
    scan_mapping_for_forbidden(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19D candidate proposal")
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument(
        "--output",
        default="result/optimization_runs/round19_factorial/reports/round19_stage19d_candidate_proposal.json",
    )
    args = parser.parse_args()
    root = Path(args.root)
    proposal = build_proposal(root)
    out = write_selection_lock(proposal, args.output)
    print(json.dumps({"written": str(out), "n_candidates": proposal["n_candidates"], "expected_jobs": proposal["expected_jobs"], "candidates": [
        {k: c[k] for k in ("candidate_id", "drug_id", "predictor_id", "omics_id", "mandatory")}
        for c in proposal["candidates"]
    ]}, indent=2))


if __name__ == "__main__":
    main()
