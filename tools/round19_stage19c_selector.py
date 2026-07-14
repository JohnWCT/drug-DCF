#!/usr/bin/env python3
"""Round 19C candidate cell selector from Stage 19B metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round19_manifest_validator import FORBIDDEN_SELECTION_COLS
from tools.round19_selection_lock import scan_mapping_for_forbidden, write_selection_lock

ROLE_FIXED = {
    "R0": ("D0", "P0"),
    "R1": ("D0", "P1"),
    "R2": ("D0", "P2"),
}

ROLE_CANDIDATES = {
    "R3": [("D1", "P0"), ("D2", "P0"), ("D3", "P0"), ("D4", "P0")],
    "R4": [("D1", "P1"), ("D2", "P1"), ("D3", "P1"), ("D4", "P1")],
    "R5": [("D2", "P2"), ("D3", "P2")],
    "R6": [("D4", "P0"), ("D4", "P1")],
}


def _sha256_file(path: Path) -> str:
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


def _assert_no_forbidden_columns(df: pd.DataFrame) -> None:
    hits = sorted(FORBIDDEN_SELECTION_COLS.intersection(df.columns))
    if hits:
        raise AssertionError(f"Forbidden selection columns present: {hits}")


def verify_stage19b_complete(
    root: Path,
    manifest_path: Path,
    *,
    expected_jobs: int,
    require_complete: bool,
) -> Tuple[int, int]:
    df = pd.read_csv(manifest_path)
    _assert_no_forbidden_columns(df)
    n_manifest = int(len(df))
    done = 0
    for _, row in df.iterrows():
        status_path = Path(str(row["result_dir"])) / "job_status.json"
        if status_path.is_file():
            st = json.loads(status_path.read_text(encoding="utf-8"))
            if st.get("status") == "done":
                done += 1
    if require_complete and done != int(expected_jobs):
        raise RuntimeError(f"Stage19B incomplete: {done}/{expected_jobs} done")
    if n_manifest != int(expected_jobs):
        raise RuntimeError(f"Manifest job count {n_manifest} != expected {expected_jobs}")
    return done, n_manifest


def load_cell_o2_o3_scores(root: Path) -> pd.DataFrame:
    """Per (drug, predictor) mean O2/O3 DrugMacro AUC and AUPRC over 3 folds."""
    metrics_path = root / "reports" / "stage19b_job_metrics.csv"
    if metrics_path.is_file():
        jobs = pd.read_csv(metrics_path)
        _assert_no_forbidden_columns(jobs)
        drug_col = "drug" if "drug" in jobs.columns else "drug_representation_id"
        pred_col = "predictor" if "predictor" in jobs.columns else "predictor_id"
        omics_col = "omics" if "omics" in jobs.columns else "omics_id"
        jobs = jobs[jobs[omics_col].isin(["O2", "O3"])].copy()
        has_auprc = "DrugMacro_AUPRC" in jobs.columns
        rows = []
        for (d, p), g in jobs.groupby([drug_col, pred_col]):
            o2 = g[g[omics_col] == "O2"]
            o3 = g[g[omics_col] == "O3"]
            if len(o2) != 3 or len(o3) != 3:
                continue
            auc_o2 = float(o2["DrugMacro_AUC"].mean())
            auc_o3 = float(o3["DrugMacro_AUC"].mean())
            if has_auprc:
                auprc_o2 = float(o2["DrugMacro_AUPRC"].mean())
                auprc_o3 = float(o3["DrugMacro_AUPRC"].mean())
                mean_auprc = (auprc_o2 + auprc_o3) / 2.0
            else:
                mean_auprc = None
            mean_auc = (auc_o2 + auc_o3) / 2.0
            rows.append(
                {
                    "drug_id": str(d),
                    "predictor_id": str(p),
                    "mean_auc_o2": auc_o2,
                    "mean_auc_o3": auc_o3,
                    "mean_auc_o2_o3": mean_auc,
                    "mean_auprc_o2_o3": mean_auprc,
                }
            )
        df = pd.DataFrame(rows)
        if has_auprc and not df.empty:
            return df
        # Fall through to val_metrics when AUPRC missing from CSV

    manifest = pd.read_csv(root / "manifests" / "stage19b_drug_predictor_manifest.csv")
    rows = []
    for _, row in manifest.iterrows():
        if str(row["omics_id"]) not in {"O2", "O3"}:
            continue
        mpath = Path(str(row["result_dir"])) / "val_metrics.json"
        if not mpath.is_file():
            continue
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
        rows.append(
            {
                "drug_id": str(row["drug_representation_id"]),
                "predictor_id": str(row["predictor_id"]),
                "omics_id": str(row["omics_id"]),
                "fold_id": int(row["fold_id"]),
                "DrugMacro_AUC": float(metrics.get("DrugMacro_AUC", float("nan"))),
                "DrugMacro_AUPRC": float(metrics.get("DrugMacro_AUPRC", float("nan"))),
            }
        )
    df = pd.DataFrame(rows)
    _assert_no_forbidden_columns(df)
    out_rows = []
    for (d, p), g in df.groupby(["drug_id", "predictor_id"]):
        o2 = g[g["omics_id"] == "O2"]
        o3 = g[g["omics_id"] == "O3"]
        if len(o2) != 3 or len(o3) != 3:
            continue
        auc_o2 = float(o2["DrugMacro_AUC"].mean())
        auc_o3 = float(o3["DrugMacro_AUC"].mean())
        auprc_o2 = float(o2["DrugMacro_AUPRC"].mean())
        auprc_o3 = float(o3["DrugMacro_AUPRC"].mean())
        out_rows.append(
            {
                "drug_id": d,
                "predictor_id": p,
                "mean_auc_o2": auc_o2,
                "mean_auc_o3": auc_o3,
                "mean_auc_o2_o3": (auc_o2 + auc_o3) / 2.0,
                "mean_auprc_o2_o3": (auprc_o2 + auprc_o3) / 2.0,
            }
        )
    return pd.DataFrame(out_rows)


def _pick_best(candidates: Sequence[Tuple[str, str]], scores: pd.DataFrame) -> Tuple[str, str, dict]:
    cand_set = set(candidates)
    subset = scores[scores.apply(lambda r: (r["drug_id"], r["predictor_id"]) in cand_set, axis=1)].copy()
    if subset.empty:
        raise RuntimeError(f"No score rows for candidates={candidates}")
    subset = subset.sort_values(
        ["mean_auc_o2_o3", "mean_auprc_o2_o3"],
        ascending=[False, False],
        na_position="last",
    )
    row = subset.iloc[0]
    return str(row["drug_id"]), str(row["predictor_id"]), row.to_dict()


def select_role_cells(scores: pd.DataFrame) -> List[dict]:
    selected: List[dict] = []
    for role, (d, p) in ROLE_FIXED.items():
        row = scores[(scores.drug_id == d) & (scores.predictor_id == p)]
        metrics = row.iloc[0].to_dict() if len(row) else {}
        selected.append(
            {
                "role": role,
                "drug_id": d,
                "predictor_id": p,
                "selection_score": metrics.get("mean_auc_o2_o3"),
                "tie_breaker_score": metrics.get("mean_auprc_o2_o3"),
            }
        )
    for role, cands in ROLE_CANDIDATES.items():
        d, p, metrics = _pick_best(cands, scores)
        selected.append(
            {
                "role": role,
                "drug_id": d,
                "predictor_id": p,
                "selection_score": metrics.get("mean_auc_o2_o3"),
                "tie_breaker_score": metrics.get("mean_auprc_o2_o3"),
            }
        )
    return selected


def deduplicate_cells(role_cells: List[dict]) -> Tuple[List[dict], List[dict]]:
    seen: Dict[Tuple[str, str], dict] = {}
    unique: List[dict] = []
    role_map: List[dict] = []
    for entry in role_cells:
        key = (entry["drug_id"], entry["predictor_id"])
        if key not in seen:
            seen[key] = entry
            unique.append(
                {
                    "drug_id": entry["drug_id"],
                    "predictor_id": entry["predictor_id"],
                    "primary_role": entry["role"],
                    "selection_score": entry.get("selection_score"),
                }
            )
            mapped = entry["role"]
        else:
            mapped = seen[key]["role"]
        role_map.append(
            {
                "role": entry["role"],
                "drug_id": entry["drug_id"],
                "predictor_id": entry["predictor_id"],
                "mapped_to_primary_role": mapped if mapped != entry["role"] else None,
            }
        )
    return unique, role_map


def pick_best_pooled_for_shuffle(unique_cells: List[dict], scores: pd.DataFrame) -> dict:
    pooled = [c for c in unique_cells if c["predictor_id"] in {"P0", "P1"}]
    if not pooled:
        raise RuntimeError("No P0/P1 cells among selected for shuffle control")
    best = None
    best_o2 = float("-inf")
    for c in pooled:
        row = scores[(scores.drug_id == c["drug_id"]) & (scores.predictor_id == c["predictor_id"])]
        if row.empty:
            continue
        o2 = float(row.iloc[0]["mean_auc_o2"])
        if o2 > best_o2:
            best_o2 = o2
            best = c
    if best is None:
        raise RuntimeError("Could not resolve best_pooled_for_shuffle")
    return {
        "drug_id": best["drug_id"],
        "predictor_id": best["predictor_id"],
        "mean_o2_drugmacro_auc": best_o2,
    }


def build_candidate_lock(
    root: Path,
    *,
    expected_jobs: int = 117,
    require_complete: bool = True,
    manifest_path: Optional[Path] = None,
    ranking_path: Optional[Path] = None,
) -> dict:
    manifest_path = manifest_path or (root / "manifests" / "stage19b_drug_predictor_manifest.csv")
    ranking_path = ranking_path or (root / "reports" / "stage19b_cell_ranking.csv")
    done, _ = verify_stage19b_complete(
        root, manifest_path, expected_jobs=expected_jobs, require_complete=require_complete
    )
    scores = load_cell_o2_o3_scores(root)
    role_cells = select_role_cells(scores)
    unique_cells, role_mapping = deduplicate_cells(role_cells)
    best_pooled = pick_best_pooled_for_shuffle(unique_cells, scores)

    manifest_sha = _sha256_file(manifest_path)
    ranking_sha = _sha256_file(ranking_path) if ranking_path.is_file() else None

    payload: Dict[str, Any] = {
        "lock_type": "stage19c_candidate_lock",
        "source_stage": "19b",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage19b_expected_jobs": int(expected_jobs),
        "stage19b_completed_jobs": int(done),
        "primary_metric": "DrugMacro_AUC",
        "tie_breaker": "DrugMacro_AUPRC",
        "role_selection_metric": "mean_of_O2_O3",
        "n_roles": len(role_cells),
        "n_selected_unique_cells": len(unique_cells),
        "selected_cells": role_cells,
        "unique_cells": unique_cells,
        "role_mapping": role_mapping,
        "best_pooled_for_shuffle": best_pooled,
        "context_shuffle_controls": {
            "atom_cell": {"drug_id": "D0", "predictor_id": "P2"},
            "pooled_cell": best_pooled,
        },
        "internal_test_used": False,
        "tcga_used": False,
        "integrated5_used": False,
        "source_manifest_sha256": manifest_sha,
        "ranking_report_sha256": ranking_sha,
        "git_commit": _git_commit(),
        "expected_core_jobs": len(unique_cells) * 2 * 3,
        "expected_control_jobs": 12,
        "expected_total_jobs": len(unique_cells) * 2 * 3 + 12,
    }
    scan_mapping_for_forbidden(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19C candidate lock from 19B metrics")
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument("--stage19b-manifest", default=None)
    parser.add_argument("--ranking", default=None)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-jobs", type=int, default=117)
    parser.add_argument("--output", required=True)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = (
        Path(args.stage19b_manifest)
        if args.stage19b_manifest
        else root / "manifests" / "stage19b_drug_predictor_manifest.csv"
    )
    ranking = Path(args.ranking) if args.ranking else root / "reports" / "stage19b_cell_ranking.csv"

    lock = build_candidate_lock(
        root,
        expected_jobs=args.expected_jobs,
        require_complete=args.require_complete,
        manifest_path=manifest,
        ranking_path=ranking,
    )
    out_path = write_selection_lock(lock, args.output)

    hash_dir = out_path.parent
    (hash_dir / "round19_stage19c_candidate_lock.sha256").write_text(
        f"{lock['source_manifest_sha256']}  stage19b_manifest\n"
        f"{lock['ranking_report_sha256'] or ''}  stage19b_ranking\n",
        encoding="utf-8",
    )

    if args.write_baseline:
        from tools.write_round19_stage19b_baseline import write_baseline

        write_baseline(root, expected_jobs=args.expected_jobs, completed_jobs=lock["stage19b_completed_jobs"])

    print(
        json.dumps(
            {"written": str(out_path), "n_selected": lock["n_selected_unique_cells"], "lock": lock},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
