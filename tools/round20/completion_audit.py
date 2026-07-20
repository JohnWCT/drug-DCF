"""Round 20 completion audit orchestrator."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from tools.round20.checkpoint_inventory import build_checkpoint_inventory
from tools.round20.lock_consistency import (
    recalculate_stage20a_decision,
    recalculate_stage20b_guardrails,
    verify_final_lock_consistency,
)
from tools.round20.result_contracts import (
    DEFAULT_RUN_ROOT,
    STAGE20A_REQUIRED,
    STAGE20B_REQUIRED,
    STAGE20C_REQUIRED,
    STAGE20D_REQUIRED_MIN,
    STAGE20E_REQUIRED,
    count_complete_jobs,
    load_json,
    load_manifest,
    scan_forbidden_selection,
    sha256_file,
    stage_dir,
    write_json,
)
from tools.round20.tcga_provenance import (
    audit_tcga_predictions,
    audit_tcga_provenance,
    export_aggregate_artifacts,
    recalculate_tcga_metrics,
)


def _git_info() -> dict:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"branch": branch, "sha": sha, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"branch": None, "sha": None, "dirty": None}


def _artifacts_present(stage_path: Path, required: tuple) -> bool:
    return all((stage_path / rel).is_file() for rel in required)


def _audit_stage20a_pairs(run_root: Path) -> dict:
    manifest = load_manifest(run_root / "stage20a_dimension/manifest.jsonl")
    pair_keys = {}
    for job in manifest:
        key = (job["split_seed"], job["fold"])
        pair_keys.setdefault(key, set()).add(job["context_id"])
    missing = [k for k, v in pair_keys.items() if v != {"C16", "C32"}]
    # split hash consistency within seed
    hash_by_seed = {}
    for job in manifest:
        hash_by_seed.setdefault(job["split_seed"], set()).add(job["split_assignment_sha256"])
    inconsistent_seeds = [s for s, hs in hash_by_seed.items() if len(hs) != 1]
    return {
        "paired_comparisons": len(pair_keys),
        "missing_pairs": len(missing),
        "split_hash_consistent": not inconsistent_seeds,
        "status": "PASS" if not missing and not inconsistent_seeds else "FAIL",
    }


def _expected_job_counts(manifest_path: Path) -> dict:
    jobs = load_manifest(manifest_path)
    train = [j for j in jobs if not j.get("skip_train")]
    skipped = [j for j in jobs if j.get("skip_train")]
    return {"total": len(jobs), "train": len(train), "skipped_reuse": len(skipped)}


def run_completion_audit(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    strict: bool = True,
    write_artifacts: bool = True,
) -> dict:
    run_root = Path(run_root)
    blocking: List[str] = []
    warnings: List[str] = []

    # Stage 20-0
    s0 = stage_dir(run_root, "20-0")
    s0_ok = (s0 / "stage20_0_go.json").is_file() and (s0 / "resolved_e3.json").is_file()
    if not s0_ok:
        blocking.append("STAGE20_0_MISSING")

    # Stage 20A
    s20a = stage_dir(run_root, "20A")
    a_counts = _expected_job_counts(s20a / "manifest.jsonl")
    a_jobs = count_complete_jobs(s20a / "jobs")
    a_complete = a_counts["train"] == a_jobs["complete"] == 30 and a_jobs["failed"] == 0
    if not _artifacts_present(s20a, STAGE20A_REQUIRED):
        blocking.append("STAGE20A_ARTIFACTS_INCOMPLETE")
    pair_audit = _audit_stage20a_pairs(run_root)
    if pair_audit["status"] != "PASS":
        blocking.append("STAGE20A_PAIR_MISMATCH")
    a_recalc = recalculate_stage20a_decision(run_root=run_root)
    if not a_recalc["stored_decision_matches_recalculation"]:
        blocking.append("STAGE20A_LOCK_MISMATCH")

    # Stage 20B
    s20b = stage_dir(run_root, "20B")
    b_counts = _expected_job_counts(s20b / "manifest.jsonl")
    b_jobs = count_complete_jobs(s20b / "jobs")
    gated_complete = b_jobs["complete"]
    b_complete = gated_complete == 15 and b_jobs["failed"] == 0
    if not _artifacts_present(s20b, STAGE20B_REQUIRED):
        blocking.append("STAGE20B_ARTIFACTS_INCOMPLETE")
    if not (s20b / "stage20b_predictor_contract.json").is_file():
        warnings.append("STAGE20B_PREDICTOR_CONTRACT_MISSING")
    if not (s20b / "gate_summary.json").is_file():
        warnings.append("STAGE20B_GATE_SUMMARY_MISSING")
    b_recalc = recalculate_stage20b_guardrails(run_root=run_root)
    if not b_recalc["stored_guardrails_match_recalculation"]:
        blocking.append("STAGE20B_GUARDRAIL_MISMATCH")

    # Stage 20C
    s20c = stage_dir(run_root, "20C")
    lock_path = s20c / "final_model_lock.json"
    c_ok = _artifacts_present(s20c, STAGE20C_REQUIRED) and load_json(lock_path).get("status") == "LOCKED"
    if not c_ok:
        blocking.append("STAGE20C_LOCK_MISSING")
    forbidden = scan_forbidden_selection(load_json(lock_path)) if c_ok else ["lock_missing"]
    if forbidden:
        blocking.append("STAGE20C_FORBIDDEN_SELECTION_INPUT")
    lock_check = verify_final_lock_consistency(run_root=run_root)
    if not lock_check["ok"]:
        blocking.append("STAGE20C_LOCK_INCONSISTENT")

    # Stage 20D
    s20d = stage_dir(run_root, "20D")
    if not _artifacts_present(s20d, STAGE20D_REQUIRED_MIN):
        blocking.append("STAGE20D_ARTIFACTS_INCOMPLETE")
    export_aggregate_artifacts(run_root=run_root)
    tcga_prov = audit_tcga_provenance(run_root=run_root)
    if tcga_prov["status"] != "PASS":
        warnings.append("TCGA_TIMING_UNVERIFIED")
    tcga_pred = audit_tcga_predictions(run_root=run_root)
    if tcga_pred["status"] != "PASS":
        blocking.append("TCGA_PREDICTION_AUDIT_FAIL")
    tcga_metrics = recalculate_tcga_metrics(run_root=run_root)
    if tcga_metrics["status"] != "PASS":
        blocking.append("TCGA_METRIC_RECALCULATION_MISMATCH")

    # Stage 20E
    s20e = stage_dir(run_root, "20E")
    e_ok = _artifacts_present(s20e, STAGE20E_REQUIRED)
    if not e_ok:
        blocking.append("STAGE20E_RELEASE_INCOMPLETE")
    release_audit = load_json(s20e / "hashes/release_audit.json") if (s20e / "hashes/release_audit.json").is_file() else {}
    if release_audit.get("status") != "PASS":
        blocking.append("STAGE20E_RELEASE_AUDIT_FAIL")

    ckpt = build_checkpoint_inventory(run_root=run_root)
    if ckpt["status"] != "PASS":
        blocking.append("CHECKPOINT_INVENTORY_INCOMPLETE")

    audit = {
        "round": 20,
        "audit_status": "PASS" if not blocking else "FAIL",
        "git": _git_info(),
        "stages": {
            "20-0": {"status": "GO" if s0_ok else "MISSING", "artifacts_complete": s0_ok},
            "20A": {
                "status": "COMPLETE" if a_complete else "PARTIAL",
                "jobs_expected": a_counts["train"],
                "jobs_complete": a_jobs["complete"],
                "jobs_failed": a_jobs["failed"],
                "pair_audit": pair_audit,
                "decision_recalc_matches": a_recalc["stored_decision_matches_recalculation"],
            },
            "20B": {
                "status": "COMPLETE" if b_complete else "PARTIAL",
                "jobs_expected": 15,
                "jobs_complete": gated_complete,
                "jobs_failed": b_jobs["failed"],
                "baseline_reused_from_20A": b_counts["skipped_reuse"] == 15,
                "guardrail_recalc_matches": b_recalc["stored_guardrails_match_recalculation"],
            },
            "20C": {"status": "LOCKED" if c_ok else "MISSING", "lock_consistency": lock_check},
            "20D": {
                "status": "COMPLETE" if tcga_pred["status"] == "PASS" else "PARTIAL",
                "provenance": tcga_prov,
                "prediction_audit": tcga_pred,
                "metrics_recalc": {"status": tcga_metrics["status"], "mismatches": tcga_metrics["mismatches"]},
            },
            "20E": {"status": release_audit.get("status", "MISSING")},
        },
        "stage20a_decision_recalc": a_recalc,
        "stage20b_guardrail_recalc": b_recalc,
        "checkpoint_inventory": ckpt,
        "blocking_errors": blocking,
        "warnings": warnings,
    }
    if write_artifacts:
        write_json(run_root / "round20_completion_audit.json", audit)
        write_json(run_root / "stage20a_dimension/stage20a_decision_recalc_audit.json", a_recalc)
        write_json(run_root / "stage20b_predictor/stage20b_guardrail_recalc_audit.json", b_recalc)
        if c_ok:
            write_json(
                run_root / "stage20c_lock/final_model_lock.sha256",
                {"sha256": sha256_file(lock_path), "path": str(lock_path)},
            )
    if strict and blocking:
        raise SystemExit(f"ROUND20_COMPLETION_AUDIT=FAIL errors={blocking}")
    return audit
