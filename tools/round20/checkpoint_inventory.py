"""Checkpoint inventory for locked Round 20 model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tools.round20.result_contracts import DEFAULT_RUN_ROOT, load_json, sha256_file


def resolve_locked_checkpoints(*, run_root: Path = DEFAULT_RUN_ROOT) -> List[dict]:
    lock = load_json(run_root / "stage20c_lock/final_model_lock.json")
    ctx = lock["selected_context"]["id"]
    cand = lock["selected_model"]["candidate_id"]
    if cand == "B_E3":
        root = run_root / "stage20a_dimension/jobs"
        pattern = "r20a__A_{ctx}_E3__ss{seed}__f{fold}__ms101"
    else:
        root = run_root / "stage20b_predictor/jobs"
        pattern = "r20b__B_GATED__{ctx}__ss{seed}__f{fold}__ms101"
    entries = []
    for seed in (52, 62, 72):
        for fold in range(5):
            job_id = pattern.format(ctx=ctx, seed=seed, fold=fold)
            ckpt = root / job_id / "best_checkpoint.pt"
            entries.append(
                {
                    "job_id": job_id,
                    "split_seed": seed,
                    "fold": fold,
                    "path": str(ckpt),
                    "exists": ckpt.is_file(),
                    "sha256": sha256_file(ckpt) if ckpt.is_file() else None,
                    "omics_dim": int(lock["selected_context"]["omics_dimension"]),
                }
            )
    return entries


def build_checkpoint_inventory(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    entries = resolve_locked_checkpoints(run_root=run_root)
    missing = [e["job_id"] for e in entries if not e["exists"]]
    return {
        "n_expected": 15,
        "n_present": sum(1 for e in entries if e["exists"]),
        "missing": missing,
        "checkpoints": entries,
        "status": "PASS" if not missing else "FAIL",
    }
