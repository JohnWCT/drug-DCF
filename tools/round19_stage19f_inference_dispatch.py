#!/usr/bin/env python3
"""GPU dispatcher for final-lock-pinned Round 19F post-hoc inference jobs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.round19_oom_runner import OOM_EXIT_CODE, _gpu_list  # noqa: E402
from tools.round19_stage19f_posthoc_manifest import (  # noqa: E402
    load_and_verify_final_lock,
    sha256_file,
)

REQUIRED_COLUMNS = {
    "job_id",
    "lock_payload_sha256",
    "checkpoint_sha256",
    "checkpoint_path",
    "checkpoint_size_bytes",
    "source_candidate_id",
    "role_aliases",
    "member_id",
    "split_seed",
    "fold_id",
    "drug_id",
    "predictor_id",
    "omics_id",
    "mode",
    "target",
    "target_path",
    "result_dir",
}


def read_and_verify_manifest(
    manifest_path: Path, final_lock_path: Path, project_root: Path
) -> List[Dict[str, str]]:
    """Reject any job not pinned exactly to the verified final lock."""
    lock = load_and_verify_final_lock(final_lock_path, project_root)
    pinned = {
        (
            str(item["source_candidate_id"]),
            str(item["member_id"]),
        ): item
        for item in lock["hashes"]["checkpoint_inventory"]
    }
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Inference manifest missing columns: {sorted(missing)}")
        jobs = list(reader)
    if not jobs:
        raise AssertionError("Inference manifest is empty")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise AssertionError("Inference manifest has duplicate job_id values")

    expected_lock_hash = lock["hashes"]["lock_payload_sha256"]
    identities = lock["_verified_checkpoint_identities"]
    for job in jobs:
        if job["mode"] not in {"infer_internal_test", "infer_tcga"}:
            raise AssertionError(f"Unsupported post-hoc mode: {job['mode']}")
        if job["mode"] == "infer_internal_test":
            if job["target"] != "internal_test" or job["target_path"]:
                raise AssertionError("Internal inference target fields are inconsistent")
        elif not job["target"] or not job["target_path"]:
            raise AssertionError("TCGA inference requires target and target_path")
        if job["lock_payload_sha256"] != expected_lock_hash:
            raise AssertionError(f"Job is not pinned to final lock: {job['job_id']}")
        key = (job["source_candidate_id"], job["member_id"])
        item = pinned.get(key)
        if item is None:
            raise AssertionError(f"Job references an unlocked checkpoint: {job['job_id']}")
        for column in ("checkpoint_path", "checkpoint_sha256"):
            if str(job[column]) != str(item[column]):
                raise AssertionError(f"{column} differs from final lock: {job['job_id']}")
        if int(job["checkpoint_size_bytes"]) != int(item["checkpoint_size_bytes"]):
            raise AssertionError(f"Checkpoint size differs from final lock: {job['job_id']}")
        for column, value in identities[job["source_candidate_id"]].items():
            if job[column] != value:
                raise AssertionError(f"{column} identity mismatch: {job['job_id']}")
    return jobs


def _nvidia_gpu_memory() -> List[Tuple[str, int, int]]:
    """Return (GPU index, total MiB, free MiB), preserving Round 19 enumeration."""
    enumerated = _gpu_list()
    if enumerated == [""]:
        return [("", 0, 0)]
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        queried = {}
        for line in output.splitlines():
            index, total, free = (part.strip() for part in line.split(","))
            queried[index] = (int(total), int(free))
        return [
            (index, queried.get(index, (0, 0))[0], queried.get(index, (0, 0))[1])
            for index in enumerated
        ]
    except (OSError, subprocess.CalledProcessError, ValueError):
        return [(index, 0, 0) for index in enumerated]


def build_gpu_slots(
    *,
    target_vram_fraction: float = 0.90,
    estimated_job_mb: int = 3500,
    reserve_mb: int = 512,
    max_jobs_per_gpu: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, int]]]:
    if not 0 < target_vram_fraction <= 1:
        raise ValueError("target_vram_fraction must be in (0, 1]")
    if estimated_job_mb <= 0:
        raise ValueError("estimated_job_mb must be positive")
    slots: List[str] = []
    plan = []
    for index, total_mb, free_mb in _nvidia_gpu_memory():
        if not index:
            count = 1
            budget_mb = 0
        else:
            budget_mb = max(
                0, min(free_mb, int(total_mb * target_vram_fraction)) - reserve_mb
            )
            count = max(1, budget_mb // estimated_job_mb)
        if max_jobs_per_gpu is not None:
            count = min(count, max(1, int(max_jobs_per_gpu)))
        count = min(int(count), 32)
        slots.extend([index] * count)
        plan.append(
            {
                "gpu": int(index) if index else -1,
                "total_mb": total_mb,
                "free_mb": free_mb,
                "packing_budget_mb": budget_mb,
                "slots": count,
            }
        )
    return slots or [""], plan


def build_inference_command(
    *,
    job: Mapping[str, str],
    pipeline: str,
    python_exe: str,
    micro_batch: int,
    settings: str,
    internal_test_path: str,
) -> List[str]:
    cmd = [
        python_exe,
        pipeline,
        "--mode",
        job["mode"],
        "--settings",
        settings,
        "--result-dir",
        job["result_dir"],
        "--checkpoint-path",
        job["checkpoint_path"],
        "--target-key",
        job["target"],
        "--drug-id",
        job["drug_id"],
        "--predictor-id",
        job["predictor_id"],
        "--omics-id",
        job["omics_id"],
        "--candidate-id",
        job["source_candidate_id"],
        "--source-candidate-id",
        job["source_candidate_id"],
        "--split-seed",
        str(int(float(job["split_seed"]))),
        "--fold-id",
        str(int(float(job["fold_id"]))),
        "--micro-batch-size",
        str(micro_batch),
    ]
    if job["mode"] == "infer_internal_test":
        cmd.extend(["--internal-test-path", internal_test_path])
    else:
        cmd.extend(["--target-path", job["target_path"]])
    return cmd


def run_job_with_oom_retry(
    *,
    job: Mapping[str, str],
    pipeline: str,
    python_exe: str,
    cuda_device: str,
    micro_batch_candidates: Sequence[int],
    max_retries: int,
    settings: str,
    internal_test_path: str,
    project_root: Path,
) -> Dict[str, object]:
    checkpoint = Path(job["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    expected_hash = job["checkpoint_sha256"]
    if sha256_file(checkpoint) != expected_hash:
        raise AssertionError(f"Checkpoint changed before dispatch: {checkpoint}")

    result_dir = Path(job["result_dir"])
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    history = []
    status: Dict[str, object] = {
        "job_id": job["job_id"],
        "status": "pending",
        "cuda_device": cuda_device,
        "result_dir": job["result_dir"],
        "checkpoint_path": job["checkpoint_path"],
        "checkpoint_sha256": expected_hash,
        "lock_payload_sha256": job["lock_payload_sha256"],
        "mode": job["mode"],
        "target": job["target"],
    }
    candidates = [int(value) for value in micro_batch_candidates]
    if not candidates:
        raise ValueError("At least one micro-batch candidate is required")

    for attempt, micro_batch in enumerate(candidates):
        if attempt > max_retries:
            break
        command = build_inference_command(
            job=job,
            pipeline=pipeline,
            python_exe=python_exe,
            micro_batch=micro_batch,
            settings=settings,
            internal_test_path=internal_test_path,
        )
        env = os.environ.copy()
        if cuda_device:
            env["CUDA_VISIBLE_DEVICES"] = cuda_device
        log_path = result_dir / f"infer_mb{micro_batch}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if sha256_file(checkpoint) != expected_hash:
            raise AssertionError(f"Inference process modified locked checkpoint: {checkpoint}")
        status.update(
            {
                "exit_code": process.returncode,
                "elapsed_sec": round(time.time() - started, 2),
                "last_log": str(log_path),
            }
        )
        if process.returncode == 0:
            status.update(
                {
                    "status": "done",
                    "successful_micro_batch": micro_batch,
                    "oom_retry_count": len(history),
                    "oom_batch_history": history,
                }
            )
            _write_status(result_dir, status)
            return status
        if process.returncode == OOM_EXIT_CODE:
            history.append(micro_batch)
            continue
        status.update(
            {
                "status": "failed",
                "detail": f"non_oom_exit_{process.returncode}",
                "oom_retry_count": len(history),
                "oom_batch_history": history,
            }
        )
        _write_status(result_dir, status)
        return status

    status.update(
        {
            "status": "oom_exhausted",
            "detail": "all permitted micro-batch attempts failed",
            "oom_retry_count": len(history),
            "oom_batch_history": history,
        }
    )
    _write_status(result_dir, status)
    return status


def _write_status(result_dir: Path, status: Mapping[str, object]) -> None:
    (result_dir / "dispatch_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )


def _write_status_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dispatch_manifest(
    *,
    manifest_path: Path,
    final_lock_path: Path,
    project_root: Path,
    pipeline: str,
    python_exe: str = sys.executable,
    execute: bool = False,
    limit: Optional[int] = None,
    micro_batch_candidates: Sequence[int] = (512, 256, 128, 64, 32),
    max_retries: int = 4,
    target_vram_fraction: float = 0.90,
    estimated_job_mb: int = 3500,
    max_jobs_per_gpu: Optional[int] = None,
    settings: str = "config/round19_factorial_settings.json",
    internal_test_path: str = (
        "result/optimization_runs/round19_factorial/splits/internal_test_split.csv"
    ),
    status_csv: Optional[Path] = None,
) -> Dict[str, object]:
    jobs = read_and_verify_manifest(manifest_path, final_lock_path, project_root)
    pending = []
    for job in jobs:
        result_dir = Path(job["result_dir"])
        if not result_dir.is_absolute():
            result_dir = project_root / result_dir
        status_path = result_dir / "dispatch_status.json"
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "done"
                and previous.get("checkpoint_sha256") == job["checkpoint_sha256"]
                and previous.get("lock_payload_sha256") == job["lock_payload_sha256"]
            ):
                continue
        pending.append(job)
    if limit is not None:
        pending = pending[: max(0, int(limit))]

    slots, gpu_plan = build_gpu_slots(
        target_vram_fraction=target_vram_fraction,
        estimated_job_mb=estimated_job_mb,
        max_jobs_per_gpu=max_jobs_per_gpu,
    )
    preview = [
        build_inference_command(
            job=job,
            pipeline=pipeline,
            python_exe=python_exe,
            micro_batch=int(micro_batch_candidates[0]),
            settings=settings,
            internal_test_path=internal_test_path,
        )
        for job in pending[:3]
    ]
    if not execute:
        return {
            "dry_run": True,
            "manifest": str(manifest_path),
            "verified_jobs": len(jobs),
            "pending_jobs": len(pending),
            "gpu_plan": gpu_plan,
            "worker_slots": len(slots),
            "command_preview": preview,
        }

    rows: List[Dict[str, object]] = []
    status_path = status_csv or manifest_path.with_name(
        manifest_path.stem + "_dispatch_status.csv"
    )

    def worker(index_job: Tuple[int, Mapping[str, str]]) -> Dict[str, object]:
        index, job = index_job
        return run_job_with_oom_retry(
            job=job,
            pipeline=pipeline,
            python_exe=python_exe,
            cuda_device=slots[index % len(slots)],
            micro_batch_candidates=micro_batch_candidates,
            max_retries=max_retries,
            settings=settings,
            internal_test_path=internal_test_path,
            project_root=project_root,
        )

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = {
            executor.submit(worker, (index, job)): job
            for index, job in enumerate(pending)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            _write_status_csv(status_path, rows)
            print(
                json.dumps(
                    {
                        "job_id": row["job_id"],
                        "status": row["status"],
                        "completed": completed,
                        "total": len(pending),
                    }
                ),
                flush=True,
            )
    summary = {
        "dry_run": False,
        "manifest": str(manifest_path),
        "n_pending": len(pending),
        "n_done": sum(row["status"] == "done" for row in rows),
        "n_failed": sum(row["status"] != "done" for row in rows),
        "gpu_plan": gpu_plan,
        "worker_slots": len(slots),
        "status_csv": str(status_path),
    }
    status_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19F post-hoc GPU dispatcher")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--pipeline", default="step1_finetune_latent_pipeline_round19.py")
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument(
        "--internal-test-path",
        default=(
            "result/optimization_runs/round19_factorial/splits/internal_test_split.csv"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--micro-batch-candidates", default="512,256,128,64,32")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--target-vram-fraction", type=float, default=0.90)
    parser.add_argument("--estimated-job-mb", type=int, default=3500)
    parser.add_argument("--max-jobs-per-gpu", type=int)
    parser.add_argument("--status-csv")
    args = parser.parse_args()
    candidates = [
        int(value)
        for value in args.micro_batch_candidates.split(",")
        if value.strip()
    ]
    summary = dispatch_manifest(
        manifest_path=Path(args.manifest),
        final_lock_path=Path(args.final_lock),
        project_root=Path(args.project_root),
        pipeline=args.pipeline,
        execute=args.execute,
        limit=args.limit,
        micro_batch_candidates=candidates,
        max_retries=args.max_retries,
        target_vram_fraction=args.target_vram_fraction,
        estimated_job_mb=args.estimated_job_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        settings=args.settings,
        internal_test_path=args.internal_test_path,
        status_csv=Path(args.status_csv) if args.status_csv else None,
    )
    print(json.dumps(summary, indent=2))
    if args.execute and summary["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
