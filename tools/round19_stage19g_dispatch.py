#!/usr/bin/env python3
"""GPU dispatcher for immutable-lock Round 19G shard jobs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.round19_oom_runner import OOM_EXIT_CODE  # noqa: E402
from tools.round19_stage19f_inference_dispatch import build_gpu_slots  # noqa: E402
from tools.round19_stage19g_executor import (  # noqa: E402
    METHODS,
    _atomic_json,
    finalize_outputs,
    smoke,
    verify_experiment_lock,
)
from tools.round19_telegram_notify import (  # noqa: E402
    notify_progress,
    notify_stage_done,
    notify_stage_fail,
    notify_stage_start,
)


def _principal_sources(verified: Mapping[str, Any]) -> list[str]:
    preferred = ("source_performance_champion", "cancer_shift_specialist")
    values = []
    roles = verified["_final"]["roles"]
    inventory_sources = {
        str(row["source_candidate_id"])
        for row in verified["_final"]["hashes"]["checkpoint_inventory"]
    }
    for role in preferred:
        requested = str(roles[role]["source_candidate_id"])
        matches = sorted(
            source for source in inventory_sources
            if source == requested or source.startswith(requested + "_")
            or requested.startswith(source + "_")
        )
        if len(matches) != 1:
            raise AssertionError(f"Principal role {role} does not resolve uniquely")
        if matches[0] not in values:
            values.append(matches[0])
    if len(values) != 2:
        raise AssertionError("Pilot requires two distinct principal locked sources")
    return values


def build_jobs(
    verified: Mapping[str, Any], *, methods: Sequence[str],
    pilot: bool = False, smoke_mode: bool = False, limit: int | None = None,
) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    principal = _principal_sources(verified) if pilot else []
    for method in methods:
        frame = verified["_manifests"][method].copy()
        if method == "routing":
            # Verify the complete pinned routing manifest, execute one all-case job.
            frame = frame.sort_values("task_id", kind="mergesort").head(1)
        elif pilot:
            frame = frame[
                frame["source_candidate_id"].isin(principal)
                & (frame["case_shard_id"] == sorted(frame["case_shard_id"].unique())[0])
            ]
        elif smoke_mode:
            frame = frame.sort_values("task_id", kind="mergesort").head(1)
        jobs.extend(frame.to_dict("records"))
    if limit is not None:
        jobs = jobs[:max(0, int(limit))]
    return jobs


def _command(
    job: Mapping[str, str], *, python_exe: str, experiment_lock: Path,
    output_root: Path, settings: Path, batch_size: int, case_limit: int | None,
) -> list[str]:
    command = [
        python_exe, "tools/round19_stage19g_executor.py",
        "--experiment-lock", str(experiment_lock),
        "--output-root", str(output_root),
        "--settings", str(settings),
        "--method", str(job["method"]),
        "--task-id", str(job["task_id"]),
        "--batch-size", str(batch_size),
        "--perturbation-batch", str(min(batch_size, 128)),
    ]
    if case_limit is not None:
        command.extend(["--case-limit", str(case_limit)])
    return command


def _status_path(output_root: Path, job: Mapping[str, str]) -> Path:
    return output_root / "shards" / job["method"] / job["task_id"] / "status.json"


def _resume_done(
    output_root: Path, job: Mapping[str, str], experiment_hash: str,
    expected_case_count: int,
) -> bool:
    path = _status_path(output_root, job)
    if not path.is_file():
        return False
    status = json.loads(path.read_text(encoding="utf-8"))
    return (
        status.get("status") == "done"
        and status.get("experiment_lock_sha256") == experiment_hash
        and status.get("checkpoint_sha256") == job["checkpoint_sha256"]
        and int(status.get("case_count", -1)) == int(expected_case_count)
        and all(Path(value).is_file() for value in status.get("outputs", []))
    )


def _run_with_retry(
    job: Mapping[str, str], *, cuda_device: str, python_exe: str,
    experiment_lock: Path, output_root: Path, settings: Path,
    batch_candidates: Sequence[int], max_retries: int, case_limit: int | None,
) -> dict[str, Any]:
    history = []
    for attempt, batch_size in enumerate(batch_candidates):
        if attempt > max_retries:
            break
        command = _command(
            job, python_exe=python_exe, experiment_lock=experiment_lock,
            output_root=output_root, settings=settings, batch_size=int(batch_size),
            case_limit=case_limit,
        )
        result_dir = _status_path(output_root, job).parent
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / f"dispatch_mb{batch_size}.log"
        env = os.environ.copy()
        if cuda_device:
            env["CUDA_VISIBLE_DEVICES"] = cuda_device
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
        row = {
            "task_id": job["task_id"], "method": job["method"],
            "source_candidate_id": job["source_candidate_id"],
            "member_id": job["member_id"], "cuda_device": cuda_device or "cpu",
            "batch_size": int(batch_size), "exit_code": process.returncode,
            "elapsed_sec": round(time.time() - started, 2), "log": str(log_path),
            "oom_batch_history": list(history),
        }
        if process.returncode == 0:
            row["status"] = "done"
            return row
        if process.returncode == OOM_EXIT_CODE:
            history.append(int(batch_size))
            continue
        row["status"] = "failed"
        return row
    return {
        "task_id": job["task_id"], "method": job["method"],
        "source_candidate_id": job["source_candidate_id"],
        "member_id": job["member_id"], "cuda_device": cuda_device or "cpu",
        "status": "oom_exhausted", "exit_code": OOM_EXIT_CODE,
        "oom_batch_history": history,
    }


def _write_dispatch_status(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


def validate_pilot(output_root: Path, sources: Sequence[str]) -> dict[str, Any]:
    """Validate 10-case, two-source, complete-member pilot invariants."""
    import numpy as np
    import pandas as pd

    atom_paths = sorted((output_root / "shards" / "occlusion").glob(
        "*/round19g_atom_occlusion.csv"
    ))
    if not atom_paths:
        raise AssertionError("pilot has no occlusion outputs")
    atom = pd.concat([pd.read_csv(path) for path in atom_paths], ignore_index=True)
    applicable = atom[atom["applicable"].astype(str).str.lower() != "false"]
    for source in sources:
        group = applicable[applicable["source_candidate_id"] == source]
        if set(group["member_id"].astype(str)) != set(REQUIRED_MEMBER_IDS):
            raise AssertionError(f"pilot {source} does not contain 15/15 members")
        if group["case_id"].nunique() != 10:
            raise AssertionError(f"pilot {source} does not contain exactly 10 cases")
        originals = group.groupby(["case_id", "member_id"])["original_probability"].nunique()
        if not originals.eq(1).all():
            raise AssertionError("pilot original probabilities are inconsistent")
        random = group[group["control_type"] == "matched_random"]
        if random.empty:
            raise AssertionError("pilot matched-random occlusion is absent")
    attention_paths = sorted((output_root / "shards" / "attention").glob(
        "*/round19g_attention_long.csv"
    ))
    attention_cases = 0
    if attention_paths:
        attention = pd.concat([pd.read_csv(path) for path in attention_paths], ignore_index=True)
        primary = attention[attention["attention_kind"] == "primary"]
        sums = primary.groupby(["candidate_id", "eval_row_id", "member_id"])["attention"].sum()
        if not np.allclose(sums, 1.0, atol=1e-5):
            raise AssertionError("pilot primary attention does not sum to one")
        if primary.groupby(["candidate_id", "eval_row_id", "member_id"])["graph_smiles"].nunique().gt(1).any():
            raise AssertionError("pilot atom mapping differs within a model/case")
        attention_probability = (
            primary.groupby(["candidate_id", "eval_row_id", "member_id"], as_index=False)
            .agg(logit=("logit", "first"))
        )
        attention_probability["attention_probability"] = 1.0 / (
            1.0 + np.exp(-attention_probability["logit"].astype(float))
        )
        baseline = (
            applicable.groupby(
                ["source_candidate_id", "eval_row_id", "member_id"], as_index=False
            )
            .agg(original_probability=("original_probability", "first"))
            .rename(columns={"source_candidate_id": "candidate_id"})
        )
        compared = attention_probability.merge(
            baseline, on=["candidate_id", "eval_row_id", "member_id"],
            how="inner", validate="one_to_one",
        )
        # These values come from separate legal batch shapes/autocast paths.
        # Forward-equivalence unit tests remain exact; this cross-run check
        # allows only the measured GPU accumulation/AMP envelope.
        if len(compared) != len(attention_probability) or not np.allclose(
            compared["attention_probability"],
            compared["original_probability"],
            atol=5e-4,
            rtol=5e-4,
        ):
            raise AssertionError(
                "pilot same-checkpoint attention/occlusion original probability mismatch"
            )
        attention_cases = int(primary["eval_row_id"].nunique())
    return {
        "pilot": "passed", "sources": list(sources), "cases_per_source": 10,
        "members_per_source": 15, "attention_cases": attention_cases,
        "checks": [
            "15/15 members", "case/atom mapping", "attention sum",
            "occlusion controls", "same-checkpoint original probability",
        ],
    }


# Imported late to keep dispatcher import light in --smoke.
from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS  # noqa: E402


def dispatch(
    *, experiment_lock: Path, output_root: Path, settings: Path,
    methods: Sequence[str], execute: bool, pilot: bool, smoke_mode: bool,
    limit: int | None, target_vram_fraction: float, estimated_job_mb: int,
    max_jobs_per_gpu: int | None, batch_candidates: Sequence[int],
    max_retries: int, python_exe: str = sys.executable,
) -> dict[str, Any]:
    verified = verify_experiment_lock(experiment_lock, project_root=PROJECT_ROOT)
    jobs = build_jobs(
        verified, methods=methods, pilot=pilot, smoke_mode=smoke_mode, limit=limit
    )
    case_limit = 10 if pilot else 1 if smoke_mode else None
    pending = []
    for job in jobs:
        expected_case_count = int(job["case_count"])
        if case_limit is not None:
            expected_case_count = min(expected_case_count, case_limit)
        if not _resume_done(
            output_root, job, verified["_file_sha256"], expected_case_count
        ):
            pending.append(job)
    slots, gpu_plan = build_gpu_slots(
        target_vram_fraction=target_vram_fraction,
        estimated_job_mb=estimated_job_mb,
        max_jobs_per_gpu=max_jobs_per_gpu,
    )
    preview = [
        _command(
            job, python_exe=python_exe, experiment_lock=experiment_lock,
            output_root=output_root, settings=settings,
            batch_size=int(batch_candidates[0]), case_limit=case_limit,
        )
        for job in pending[:3]
    ]
    if not execute:
        return {
            "dry_run": True, "verified_jobs": len(jobs), "pending_jobs": len(pending),
            "pilot": pilot, "smoke": smoke_mode, "gpu_plan": gpu_plan,
            "worker_slots": len(slots), "command_preview": preview,
        }
    stage = "19g-pilot" if pilot else "19g-smoke" if smoke_mode else "19g-formal"
    notify_stage_start(stage, f"jobs={len(pending)}")
    rows: list[dict[str, Any]] = []
    status_csv = output_root / f"{stage}_dispatch_status.csv"
    try:
        def worker(index_job):
            index, job = index_job
            return _run_with_retry(
                job, cuda_device=slots[index % len(slots)], python_exe=python_exe,
                experiment_lock=experiment_lock, output_root=output_root,
                settings=settings, batch_candidates=batch_candidates,
                max_retries=max_retries, case_limit=case_limit,
            )

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = {
                pool.submit(worker, pair): pair[1]
                for pair in enumerate(pending)
            }
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                _write_dispatch_status(status_csv, rows)
                print(json.dumps({
                    "task_id": row["task_id"], "status": row["status"],
                    "completed": completed, "total": len(pending),
                }), flush=True)
                if completed % 25 == 0:
                    notify_progress(stage)
        failed = [row for row in rows if row["status"] != "done"]
        result: dict[str, Any] = {
            "dry_run": False, "jobs": len(jobs), "pending": len(pending),
            "done": len(rows) - len(failed), "failed": len(failed),
            "gpu_plan": gpu_plan, "worker_slots": len(slots),
            "status_csv": str(status_csv),
        }
        if failed:
            raise RuntimeError(f"{len(failed)} Round19G jobs failed")
        if pilot:
            result["pilot_validation"] = validate_pilot(
                output_root, _principal_sources(verified)
            )
        elif not smoke_mode and set(methods) == set(METHODS):
            result["finalize"] = finalize_outputs(
                experiment_lock=experiment_lock, output_root=output_root
            )
        _atomic_json(status_csv.with_suffix(".summary.json"), result)
        notify_stage_done(stage)
        return result
    except Exception as exc:
        notify_stage_fail(stage, str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19G locked GPU dispatcher")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--experiment-lock")
    parser.add_argument("--output-root", default=(
        "result/optimization_runs/round19_factorial/stage19g"
    ))
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument("--methods", default="attention,occlusion,omics,routing")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-vram-fraction", type=float, default=0.90)
    parser.add_argument("--estimated-job-mb", type=int, default=3500)
    parser.add_argument("--max-jobs-per-gpu", type=int)
    parser.add_argument("--batch-candidates", default="128,64,32,16")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--real-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.smoke and not args.experiment_lock:
        print(json.dumps(smoke(args.real_checkpoint), indent=2))
        return
    if not args.experiment_lock:
        parser.error("--experiment-lock is required for pilot/formal or locked smoke")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    if not methods or not set(methods) <= set(METHODS):
        parser.error(f"--methods must be a subset of {METHODS}")
    if args.formal and not args.execute:
        # A formal dry-run remains useful and cannot mutate analysis outputs.
        pass
    result = dispatch(
        experiment_lock=Path(args.experiment_lock),
        output_root=Path(args.output_root), settings=Path(args.settings),
        methods=methods, execute=args.execute, pilot=args.pilot,
        smoke_mode=args.smoke, limit=args.limit,
        target_vram_fraction=args.target_vram_fraction,
        estimated_job_mb=args.estimated_job_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        batch_candidates=[
            int(value) for value in args.batch_candidates.split(",") if value.strip()
        ],
        max_retries=args.max_retries,
    )
    print(json.dumps(result, indent=2, default=str))
    if args.execute and result.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
