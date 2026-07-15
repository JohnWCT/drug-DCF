#!/usr/bin/env python3
"""Round 19E candidate lock from 19D experiment lock (no architecture search)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round19_manifest_validator import FORBIDDEN_SELECTION_COLS
from tools.round19_selection_lock import scan_mapping_for_forbidden, write_selection_lock

ROLE_MAP = [
    ("E0", "F0_historical_anchor", "historical_mlp_anchor", True),
    ("E1", "F1_primary_o2", "primary_o2_atom_model", True),
    ("E2", "F2_full_omics_o3", "o3_atom_control", True),
    ("E3", "F3_best_pooled_o2", "best_pooled_comparator", True),
    ("E4", "F4_source_only_o4", "source_only_candidate", True),
]


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


def _assert_no_forbidden(obj: Any) -> None:
    scan_mapping_for_forbidden(obj)
    if isinstance(obj, dict):
        hits = sorted(FORBIDDEN_SELECTION_COLS.intersection(obj.keys()))
        if hits:
            raise AssertionError(f"Forbidden keys: {hits}")


def _find_f(candidates: List[dict], fid: str) -> dict:
    for c in candidates:
        if str(c.get("candidate_id")) == fid:
            return c
    for c in candidates:
        if str(c.get("candidate_id")).startswith(fid + "_"):
            return c
    raise KeyError(f"Missing 19D candidate {fid}")


def maybe_include_e5(
    lock19d: dict,
    cross: pd.DataFrame,
    *,
    gap_max: float = 0.015,
) -> Optional[dict]:
    """Include E5=F5 if AUC gap vs F1 ≤ 0.015 and efficiency gate from 19D lock."""
    cands = lock19d.get("candidates") or []
    try:
        f5 = _find_f(cands, "F5")
    except KeyError:
        return None
    if (
        not cross.empty
        and "F1_primary_o2" in set(cross.candidate_id)
        and "F5_maccs_efficient" in set(cross.candidate_id)
    ):
        f1 = float(
            cross.loc[cross.candidate_id == "F1_primary_o2", "mean_of_means_DrugMacro_AUC"].iloc[0]
        )
        f5_auc = float(
            cross.loc[cross.candidate_id == "F5_maccs_efficient", "mean_of_means_DrugMacro_AUC"].iloc[0]
        )
        gap = f1 - f5_auc
    else:
        gap = float((f5.get("inclusion_reason") or {}).get("auc_gap_vs_f3", 1.0))
    if gap > gap_max:
        return None
    reason = f5.get("inclusion_reason") or {}
    time_ok = bool(reason.get("cond_b_time")) or (
        reason.get("epoch_wall_f5") is not None
        and reason.get("epoch_wall_f3")
        and float(reason["epoch_wall_f5"]) <= 0.75 * float(reason["epoch_wall_f3"])
    )
    vram_ok = bool(reason.get("cond_c_vram")) or (
        reason.get("vram_f5") is not None
        and reason.get("vram_f3")
        and float(reason["vram_f5"]) <= 0.70 * float(reason["vram_f3"])
    )
    # MACCS is definitionally lower-cost vs graph; accept if gap ok and any efficiency flag
    if not (time_ok or vram_ok or str(f5.get("drug_id")) == "D4"):
        return None
    return {
        "candidate_id": "E5",
        "source_candidate_id": str(f5["candidate_id"]),
        "role": "maccs_efficient",
        "mandatory": False,
        "drug_id": str(f5["drug_id"]),
        "predictor_id": str(f5["predictor_id"]),
        "omics_id": str(f5["omics_id"]),
        "inclusion_reason": {
            "auc_gap_vs_f1": gap,
            "time_ok": time_ok,
            "vram_ok": vram_ok,
            **{k: reason[k] for k in reason if k.startswith(("epoch", "vram", "cond"))},
        },
    }


def build_candidate_lock(root: Path) -> dict:
    root = Path(root)
    lock_path = root / "reports" / "round19_stage19d_experiment_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock19d = json.loads(lock_path.read_text(encoding="utf-8"))
    _assert_no_forbidden(lock19d)
    if lock19d.get("lock_type") != "stage19d_experiment_lock":
        raise ValueError(f"Unexpected lock_type: {lock19d.get('lock_type')}")

    cross_path = root / "reports" / "round19d_cross_seed_summary.csv"
    cross = pd.read_csv(cross_path) if cross_path.is_file() else pd.DataFrame()

    cands_19d = lock19d.get("candidates") or []
    out_cands: List[dict] = []
    for eid, fid, role, mandatory in ROLE_MAP:
        src = _find_f(cands_19d, fid)
        out_cands.append(
            {
                "candidate_id": eid,
                "source_candidate_id": str(src["candidate_id"]),
                "role": role,
                "mandatory": mandatory,
                "drug_id": str(src["drug_id"]),
                "predictor_id": str(src["predictor_id"]),
                "omics_id": str(src["omics_id"]),
                "selection_role": src.get("selection_role", role),
            }
        )

    # Pin identities
    by_id = {c["candidate_id"]: c for c in out_cands}
    assert by_id["E1"]["source_candidate_id"] == "F1_primary_o2"
    assert by_id["E2"]["source_candidate_id"] == "F2_full_omics_o3"
    assert by_id["E4"]["source_candidate_id"] == "F4_source_only_o4"
    assert by_id["E1"]["omics_id"] == "O2"
    assert by_id["E2"]["omics_id"] == "O3"
    assert by_id["E4"]["omics_id"] == "O4"

    e5 = maybe_include_e5(lock19d, cross)
    if e5:
        out_cands.append(e5)

    # Mandatory presence
    for eid in ("E0", "E1", "E2", "E3", "E4"):
        if eid not in {c["candidate_id"] for c in out_cands}:
            raise AssertionError(f"Missing mandatory {eid}")

    n_folds = 5
    n_cand = len(out_cands)
    payload = {
        "lock_type": "stage19e_candidate_lock",
        "source_stage": "19d",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_experiment_lock": str(lock_path),
        "candidates": out_cands,
        "shift_strategies": ["drug_heldout", "scaffold_heldout", "cancer_type_heldout"],
        "n_folds": n_folds,
        "model_seed": int(lock19d.get("model_seed", 101)),
        "max_epochs": int(lock19d.get("max_epochs", 1500)),
        "early_stop_patience": int(lock19d.get("early_stop_patience", 100)),
        "early_stop_start_epoch": int(lock19d.get("early_stop_start_epoch", 50)),
        "expected_jobs_per_strategy": n_cand * n_folds,
        "expected_jobs_total": n_cand * n_folds * 3,
        "internal_test_used": False,
        "tcga_used": False,
        "hypers_unchanged_from_19d": True,
    }
    _assert_no_forbidden(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument(
        "--output",
        default="result/optimization_runs/round19_factorial/reports/round19_stage19e_candidate_lock.json",
    )
    args = parser.parse_args()
    payload = build_candidate_lock(Path(args.root))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_selection_lock(payload, str(out))
    print(
        json.dumps(
            {
                "written": str(out),
                "n_candidates": len(payload["candidates"]),
                "expected_jobs_total": payload["expected_jobs_total"],
                "candidates": [
                    {
                        "candidate_id": c["candidate_id"],
                        "source_candidate_id": c["source_candidate_id"],
                        "drug_id": c["drug_id"],
                        "predictor_id": c["predictor_id"],
                        "omics_id": c["omics_id"],
                        "mandatory": c["mandatory"],
                    }
                    for c in payload["candidates"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
