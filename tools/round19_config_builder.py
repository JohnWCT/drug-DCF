#!/usr/bin/env python3
"""Round 19 config / stage setup helpers."""
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

from tools.round19_context_controls import shuffle_seeds_for_fold
from tools.round19_cv_splits import link_or_reuse_round18_eligible, link_or_reuse_round18_splits
from tools.round19_feature_builder import OMICS_ALIAS, build_round19_feature_set
from tools.round19_fusion_models import COMPATIBLE_CELLS, assert_compatible
from tools.round19_graph_features import (
    BOND_FEATURE_DIM,
    cache_metadata,
    ensure_cache_dir,
)
from tools.round19_manifest_validator import assert_expected_job_count, validate_compatible_manifest
from tools.round19_oom_runner import REQUIRED_JOB_METADATA, assert_job_metadata


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_git_baseline(outdir: Path) -> Dict[str, Any]:
    meta_dir = Path(outdir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    def _run(cmd: List[str]) -> str:
        full = ["git", "-c", "safe.directory=*"] + cmd[1:] if cmd and cmd[0] == "git" else cmd
        return subprocess.check_output(full, cwd=str(Path.cwd()), text=True).strip()

    head = os.environ.get("ROUND19_GIT_HEAD", "").strip()
    origin = os.environ.get("ROUND19_GIT_ORIGIN_MAIN", "").strip()
    dirty = os.environ.get("ROUND19_GIT_STATUS", "")
    try:
        if not head:
            head = _run(["git", "rev-parse", "HEAD"])
        if not origin:
            try:
                origin = _run(["git", "rev-parse", "origin/main"])
            except subprocess.CalledProcessError:
                origin = ""
        if dirty == "":
            dirty = _run(["git", "status", "--porcelain"])
    except subprocess.CalledProcessError as exc:
        if not head:
            head_file = Path(".git/HEAD")
            if head_file.is_file():
                ref = head_file.read_text().strip()
                if ref.startswith("ref:"):
                    ref_path = Path(".git") / ref.split(" ", 1)[1].strip()
                    head = ref_path.read_text().strip() if ref_path.is_file() else "UNKNOWN"
                else:
                    head = ref
            else:
                head = "UNKNOWN"
        dirty = dirty or f"git_unavailable: {exc}"
    payload = {
        "round18e_commit": head,
        "round19_start_commit": head,
        "origin_main": origin,
        "head_equals_origin_main": bool(origin) and head == origin,
        "working_tree_clean": dirty == "",
        "round18e_external_success": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dirty_sample": dirty.splitlines()[:20],
    }
    path = meta_dir / "round19_baseline_git.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _drug_fields(settings: dict, drug_id: str) -> Dict[str, Any]:
    cfg = settings["drug_reps"][drug_id]
    enc = cfg["type"]
    if enc == "maccs":
        return {
            "encoder_type": "maccs",
            "node_hidden_dim": None,
            "graph_output_dim": int(cfg.get("output_dim", 64)),
            "edge_features": False,
            "edge_feature_schema": None,
            "edge_dim": None,
            "has_graph": False,
        }
    edge = bool(cfg.get("edge_features"))
    return {
        "encoder_type": enc,
        "node_hidden_dim": int(cfg["node_hidden_dim"]),
        "graph_output_dim": int(cfg["graph_output_dim"]),
        "edge_features": edge,
        "edge_feature_schema": "bond_v1" if edge else None,
        "edge_dim": int(cfg.get("edge_dim", BOND_FEATURE_DIM)) if edge else None,
        "has_graph": True,
    }


def build_stage19b_manifest(
    settings: dict,
    outdir: str,
    *,
    omics_ids: Optional[List[str]] = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """13 compatible cells × anchor omics × folds (default O1/O2/O3 × 3 = 117)."""
    root = Path(outdir)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    omics_ids = omics_ids or list(settings.get("stage19b_omics_anchors") or ["O1", "O2", "O3"])
    split_seed = int(settings.get("screening_split_seed", 42))
    model_seed = int(settings.get("model_seed", 101))
    rows = []
    for drug_id, pred_id in COMPATIBLE_CELLS:
        assert_compatible(drug_id, pred_id)
        dfields = _drug_fields(settings, drug_id)
        for omics_id in omics_ids:
            feature_dir = str(
                Path(settings.get("round19_feature_out_root", root / "features"))
                / OMICS_ALIAS[omics_id]
            )
            for fold_id in range(int(n_folds)):
                job_id = f"{drug_id}__{pred_id}__{omics_id}__fold{fold_id}"
                row = {
                    "job_id": job_id,
                    "drug_representation_id": drug_id,
                    "predictor_id": pred_id,
                    "omics_id": omics_id,
                    "omics_display_name": OMICS_ALIAS[omics_id],
                    "fold_id": fold_id,
                    "split_strategy": "modelid_grouped_screening_3fold",
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "feature_dir": feature_dir,
                    "result_dir": str(root / "stage19b" / job_id),
                    **dfields,
                }
                assert_job_metadata(row)
                rows.append(row)
    df = pd.DataFrame(rows)
    validate_compatible_manifest(df)
    bad = df[(df.drug_representation_id == "D1") & (df.predictor_id == "P2")]
    if len(bad):
        raise AssertionError(f"D1×P2 must be 0, got {len(bad)}")
    bad = df[(df.drug_representation_id == "D4") & (df.predictor_id == "P2")]
    if len(bad):
        raise AssertionError(f"D4×P2 must be 0, got {len(bad)}")
    if set(omics_ids) == {"O1", "O2", "O3"}:
        for d, p in COMPATIBLE_CELLS:
            n = int(len(df[(df.drug_representation_id == d) & (df.predictor_id == p)]))
            if n != 9:
                raise AssertionError(f"{d}×{p} expected 9 jobs (3 omics × 3 folds), got {n}")
    if not df["job_id"].is_unique:
        raise AssertionError("job_id not unique")
    if not df["result_dir"].is_unique:
        raise AssertionError("result_dir not unique")
    if set(df["fold_id"].astype(int)) != {0, 1, 2}:
        raise AssertionError(f"fold IDs must be {{0,1,2}}, got {set(df['fold_id'])}")
    expected = len(COMPATIBLE_CELLS) * len(omics_ids) * int(n_folds)
    assert_expected_job_count(df, expected, label="stage19b")
    path = manifests / "stage19b_drug_predictor_manifest.csv"
    df.to_csv(path, index=False)
    return df


def _load_candidate_lock(path: str) -> dict:
    payload = _load_json(Path(path))
    if payload.get("lock_type") != "stage19c_candidate_lock":
        raise ValueError(f"Expected stage19c_candidate_lock, got {payload.get('lock_type')}")
    return payload


def _feature_dir_for_omics(settings: dict, root: Path, omics_id: str) -> str:
    feature_root = Path(settings.get("round19_feature_out_root", root / "features"))
    return str(feature_root / OMICS_ALIAS[omics_id])


def _base_manifest_row(
    settings: dict,
    root: Path,
    *,
    drug_id: str,
    pred_id: str,
    omics_id: str,
    fold_id: int,
    split_seed: int,
    model_seed: int,
    role: Optional[str] = None,
    control_type: str = "none",
    context_control: str = "none",
    train_shuffle_seed: Optional[int] = None,
    validation_shuffle_seed: Optional[int] = None,
) -> dict:
    dfields = _drug_fields(settings, drug_id)
    suffix = ""
    if control_type == "context_shuffle":
        suffix = "__ctx_shuffle"
    job_id = f"{drug_id}__{pred_id}__{omics_id}__fold{fold_id}{suffix}"
    row = {
        "job_id": job_id,
        "drug_representation_id": drug_id,
        "predictor_id": pred_id,
        "drug_id": drug_id,
        "omics_id": omics_id,
        "omics_display_name": OMICS_ALIAS[omics_id],
        "fold_id": fold_id,
        "control_type": control_type,
        "context_control": context_control,
        "shuffle_unit": "ModelID" if control_type == "context_shuffle" else "",
        "shuffle_scope": "within_partition" if control_type == "context_shuffle" else "",
        "train_shuffle_seed": train_shuffle_seed if train_shuffle_seed is not None else "",
        "validation_shuffle_seed": validation_shuffle_seed if validation_shuffle_seed is not None else "",
        "split_strategy": "modelid_grouped_screening_3fold",
        "split_seed": split_seed,
        "model_seed": model_seed,
        "feature_dir": _feature_dir_for_omics(settings, root, omics_id),
        "result_dir": str(root / "stage19c" / job_id),
        "role": role or "",
        **dfields,
    }
    assert_job_metadata(row)
    return row


def build_stage19c_manifest(
    settings: dict,
    outdir: str,
    candidate_lock: dict,
    *,
    include_context_controls: bool = True,
) -> pd.DataFrame:
    """Build Stage 19C manifest: selected cells × O0/O4 + optional shuffle controls."""
    root = Path(outdir)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    split_seed = int(settings.get("screening_split_seed", 42))
    model_seed = int(settings.get("model_seed", 101))

    unique_cells = candidate_lock.get("unique_cells") or candidate_lock.get("selected_cells")
    if not unique_cells:
        raise ValueError("candidate lock missing unique_cells")

    seen = set()
    cells: List[dict] = []
    for c in unique_cells:
        d = str(c["drug_id"])
        p = str(c["predictor_id"])
        key = (d, p)
        if key in seen:
            continue
        seen.add(key)
        cells.append(
            {
                "drug_id": d,
                "predictor_id": p,
                "role": c.get("primary_role") or c.get("role") or "",
            }
        )

    rows: List[dict] = []
    for cell in cells:
        for omics_id in ("O0", "O4"):
            for fold_id in range(3):
                rows.append(
                    _base_manifest_row(
                        settings,
                        root,
                        drug_id=cell["drug_id"],
                        pred_id=cell["predictor_id"],
                        omics_id=omics_id,
                        fold_id=fold_id,
                        split_seed=split_seed,
                        model_seed=model_seed,
                        role=str(cell.get("role") or ""),
                        control_type="none",
                        context_control="none",
                    )
                )

    if include_context_controls:
        controls = candidate_lock.get("context_shuffle_controls") or {}
        atom = controls.get("atom_cell") or {"drug_id": "D0", "predictor_id": "P2"}
        pooled = controls.get("pooled_cell") or candidate_lock.get("best_pooled_for_shuffle") or {}
        control_cells = [
            (str(atom["drug_id"]), str(atom["predictor_id"])),
            (str(pooled["drug_id"]), str(pooled["predictor_id"])),
        ]
        for drug_id, pred_id in control_cells:
            for omics_id in ("O2", "O3"):
                for fold_id in range(3):
                    train_seed, val_seed = shuffle_seeds_for_fold(fold_id)
                    rows.append(
                        _base_manifest_row(
                            settings,
                            root,
                            drug_id=drug_id,
                            pred_id=pred_id,
                            omics_id=omics_id,
                            fold_id=fold_id,
                            split_seed=split_seed,
                            model_seed=model_seed,
                            control_type="context_shuffle",
                            context_control="shuffled",
                            train_shuffle_seed=train_seed,
                            validation_shuffle_seed=val_seed,
                        )
                    )

    df = pd.DataFrame(rows)
    validate_compatible_manifest(df)
    if not df["job_id"].is_unique:
        raise AssertionError("job_id not unique")
    if not df["result_dir"].is_unique:
        raise AssertionError("result_dir not unique")

    n_selected = len(cells)
    expected_core = n_selected * 2 * 3
    core = df[df["control_type"] == "none"]
    ctrl = df[df["control_type"] == "context_shuffle"]
    assert_expected_job_count(core, expected_core, label="stage19c_core")
    if include_context_controls:
        assert_expected_job_count(ctrl, 12, label="stage19c_controls")
        assert_expected_job_count(df, expected_core + 12, label="stage19c_total")
    else:
        assert_expected_job_count(df, expected_core, label="stage19c_total")

    path = manifests / "stage19c_manifest.csv"
    df.to_csv(path, index=False)
    return df


def build_stage19a(settings: dict, outdir: str) -> Dict[str, Any]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    git_meta = write_git_baseline(root)

    feature_root = Path(settings["feature_root"]) / settings["feature_model_key"]
    feat_rep = build_round19_feature_set(
        feature_root=str(feature_root),
        out_root=settings.get("round19_feature_out_root", str(root / "features")),
    )

    round18_root = settings.get(
        "round18_population_root", "result/optimization_runs/round18_architecture"
    )
    eligible_path = link_or_reuse_round18_eligible(round18_root, str(root))
    split_map = link_or_reuse_round18_splits(round18_root, str(root))

    cache_root = root / "cache"
    gin_meta = cache_metadata(
        encoder_type="gin",
        atom_feature_dim=78,
        bond_feature_dim=None,
        cache_version="round19_gin_atom78_v1",
    )
    gine_meta = cache_metadata(
        encoder_type="gine",
        atom_feature_dim=78,
        bond_feature_dim=BOND_FEATURE_DIM,
        cache_version="round19_gine_v1",
    )
    gin_cache = ensure_cache_dir(cache_root, "gin_atom78_v1", gin_meta)
    gine_cache = ensure_cache_dir(cache_root, "gine_atom78_bond_v1", gine_meta)

    cells = [tuple(x) for x in settings.get("compatible_cells", COMPATIBLE_CELLS)]
    for d, p in cells:
        assert_compatible(d, p)
    if sorted(cells) != sorted(COMPATIBLE_CELLS):
        raise RuntimeError("settings.compatible_cells != code COMPATIBLE_CELLS")

    manifest19b = build_stage19b_manifest(settings, str(root))

    report = {
        "stage": "19a",
        "git": git_meta,
        "features": feat_rep,
        "eligible_path": eligible_path,
        "splits": split_map,
        "cache": {"gin": str(gin_cache), "gine": str(gine_cache)},
        "n_compatible_cells": len(COMPATIBLE_CELLS),
        "stage19b_manifest_jobs": int(len(manifest19b)),
        "smoke_cells": settings.get("stage19a_smoke_cells", []),
        "required_job_metadata": REQUIRED_JOB_METADATA,
    }
    (root / "reports").mkdir(parents=True, exist_ok=True)
    path = root / "reports" / "round19a_setup_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 config builder")
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--stage", default="19a", choices=["19a", "19b_manifest", "19c"])
    parser.add_argument("--candidate-lock", default=None)
    parser.add_argument("--include-context-controls", action="store_true")
    args = parser.parse_args()
    settings = _load_json(Path(args.settings))
    outdir = args.outdir or settings.get("outdir", "result/optimization_runs/round19_factorial")
    if args.stage == "19a":
        rep = build_stage19a(settings, outdir)
        print(json.dumps(rep, indent=2, default=str))
    elif args.stage == "19b_manifest":
        df = build_stage19b_manifest(settings, outdir)
        print(json.dumps({"n_jobs": int(len(df)), "path": "manifests/stage19b_drug_predictor_manifest.csv"}))
    elif args.stage == "19c":
        if not args.candidate_lock:
            raise SystemExit("--stage 19c requires --candidate-lock")
        lock = _load_candidate_lock(args.candidate_lock)
        df = build_stage19c_manifest(
            settings,
            outdir,
            lock,
            include_context_controls=args.include_context_controls,
        )
        print(
            json.dumps(
                {
                    "n_jobs": int(len(df)),
                    "n_core": int(len(df[df.control_type == "none"])),
                    "n_controls": int(len(df[df.control_type == "context_shuffle"])),
                    "path": "manifests/stage19c_manifest.csv",
                }
            )
        )
    else:
        raise SystemExit(f"Unsupported stage {args.stage}")


if __name__ == "__main__":
    main()
