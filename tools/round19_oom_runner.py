"""Round 19 process-level OOM retry + multi-slot GPU dispatch."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_oom_runner import (  # noqa: F401
    compute_accumulation_steps,
    probe_micro_batch,
    write_resource_metadata,
)

OOM_EXIT_CODE = 42

REQUIRED_JOB_METADATA = [
    "omics_id",
    "drug_representation_id",
    "predictor_id",
    "node_hidden_dim",
    "graph_output_dim",
    "edge_feature_schema",
    "split_strategy",
    "split_seed",
]


def assert_job_metadata(job: dict) -> None:
    missing = [k for k in REQUIRED_JOB_METADATA if k not in job]
    if missing:
        raise KeyError(f"Round19 job missing metadata: {missing}")


def _read_manifest(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_status_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gpu_list() -> List[str]:
    try:
        import torch

        n = int(torch.cuda.device_count())
        if n > 0:
            return [str(i) for i in range(n)]
    except Exception:  # noqa: BLE001
        pass
    return [""]


def recommend_jobs_per_gpu(
    *,
    target_util_frac: float = 0.85,
    reserve_mb: int = 2048,
    est_job_mb: int = 3500,
) -> int:
    """Estimate safe concurrent jobs on the first GPU from free VRAM."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 1
        free, total = torch.cuda.mem_get_info(0)
        free_mb = int(free / (1024**2))
        total_mb = int(total / (1024**2))
        budget = max(0, min(free_mb, int(total_mb * target_util_frac)) - int(reserve_mb))
        n = max(1, budget // max(1, int(est_job_mb)))
        return int(min(n, 16))
    except Exception:  # noqa: BLE001
        return 1


def _build_round19_cmd(
    *,
    pipeline: str,
    job: dict,
    micro_batch: int,
    accum: int,
    python_exe: str,
    settings: str,
    response_path: str,
    split_assignment: str,
    internal_test_path: str,
) -> List[str]:
    cmd = [
        python_exe,
        pipeline,
        "--mode",
        "train_fold",
        "--settings",
        settings,
        "--result-dir",
        str(job["result_dir"]),
        "--response-path",
        response_path,
        "--split-assignment",
        str(job.get("split_assignment_path") or split_assignment),
        "--internal-test-path",
        internal_test_path,
        "--drug-id",
        str(job.get("drug_representation_id") or job.get("drug_id")),
        "--predictor-id",
        str(job["predictor_id"]),
        "--omics-id",
        str(job["omics_id"]),
        "--fold-id",
        str(int(float(job["fold_id"]))),
        "--model-seed",
        str(int(float(job.get("model_seed", 101)))),
        "--micro-batch-size",
        str(int(micro_batch)),
        "--accumulation-steps",
        str(int(accum)),
    ]
    if job.get("max_epochs"):
        cmd.extend(["--max-epochs", str(int(float(job["max_epochs"])))])
    if job.get("early_stop_patience"):
        cmd.extend(["--early-stop-patience", str(int(float(job["early_stop_patience"])))])
    if job.get("early_stop_start_epoch"):
        cmd.extend(["--early-stop-start-epoch", str(int(float(job["early_stop_start_epoch"])))])
    control_type = str(job.get("control_type") or "none")
    if control_type and control_type != "none":
        cmd.extend(["--control-type", control_type])
    if job.get("train_shuffle_seed") not in (None, ""):
        cmd.extend(["--train-shuffle-seed", str(int(float(job["train_shuffle_seed"])))])
    if job.get("validation_shuffle_seed") not in (None, ""):
        cmd.extend(["--val-shuffle-seed", str(int(float(job["validation_shuffle_seed"])))])
    return cmd


def run_single_job_with_oom_retry(
    *,
    pipeline: str,
    job: dict,
    micro_batch_candidates: Sequence[int],
    target_effective_batch: int,
    max_retries: int,
    cuda_device: str,
    settings: str,
    response_path: str,
    split_assignment: str,
    internal_test_path: str,
    python_exe: str = sys.executable,
) -> dict:
    """Process-level OOM retry: exit 42 -> smaller micro-batch -> fresh process."""
    history: List[int] = []
    requested = int(job.get("requested_micro_batch") or micro_batch_candidates[0])
    candidates = [c for c in micro_batch_candidates if int(c) <= requested]
    if not candidates:
        candidates = list(micro_batch_candidates)

    result_dir = Path(str(job["result_dir"]))
    result_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "job_id": job.get("job_id", ""),
        "status": "pending",
        "requested_micro_batch": requested,
        "successful_micro_batch": -1,
        "oom_retry_count": 0,
        "oom_batch_history": [],
        "exit_code": None,
        "detail": "",
        "result_dir": str(result_dir),
        "cuda_device": cuda_device,
        "drug_representation_id": job.get("drug_representation_id"),
        "predictor_id": job.get("predictor_id"),
        "omics_id": job.get("omics_id"),
        "fold_id": job.get("fold_id"),
    }

    for attempt, micro_batch in enumerate(candidates):
        if attempt > max_retries:
            break
        accum = compute_accumulation_steps(target_effective_batch, int(micro_batch))
        cmd = _build_round19_cmd(
            pipeline=pipeline,
            job=job,
            micro_batch=int(micro_batch),
            accum=accum,
            python_exe=python_exe,
            settings=settings,
            response_path=response_path,
            split_assignment=split_assignment,
            internal_test_path=internal_test_path,
        )
        env = os.environ.copy()
        if cuda_device != "":
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        log_path = result_dir / f"train_mb{micro_batch}.log"
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        elapsed = time.time() - t0
        status["exit_code"] = proc.returncode
        status["last_log"] = str(log_path)
        status["elapsed_sec"] = round(elapsed, 2)

        if proc.returncode == 0:
            status["status"] = "done"
            status["successful_micro_batch"] = int(micro_batch)
            status["gradient_accumulation_steps"] = accum
            status["effective_batch"] = int(micro_batch) * int(accum)
            status["oom_retry_count"] = len(history)
            status["oom_batch_history"] = list(history)
            (result_dir / "job_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            return status

        if proc.returncode == OOM_EXIT_CODE:
            history.append(int(micro_batch))
            status["status"] = "oom_retry"
            status["oom_retry_count"] = len(history)
            status["oom_batch_history"] = list(history)
            ckpt = result_dir / "checkpoint.pt"
            if ckpt.exists():
                ckpt.unlink()
            continue

        status["status"] = "failed"
        status["detail"] = f"non_oom_exit_{proc.returncode}"
        status["oom_batch_history"] = list(history)
        (result_dir / "job_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    status["status"] = "oom_exhausted"
    status["detail"] = "all micro-batch candidates failed"
    status["oom_retry_count"] = len(history)
    status["oom_batch_history"] = list(history)
    (result_dir / "job_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def dispatch_manifest(
    *,
    manifest_path: str,
    pipeline: str = "step1_finetune_latent_pipeline_round19.py",
    max_jobs_per_gpu: Optional[int] = None,
    auto_pack: bool = True,
    micro_batch_candidates: Optional[Sequence[int]] = None,
    target_effective_batch: int = 1024,
    max_retries: int = 4,
    status_csv: Optional[str] = None,
    limit: Optional[int] = None,
    job_filter: Optional[str] = None,
    settings: str = "config/round19_factorial_settings.json",
    response_path: str = "result/optimization_runs/round19_factorial/data/round19_eligible_response.csv",
    split_assignment: str = "result/optimization_runs/round19_factorial/splits/screening_3fold_assignments.csv",
    internal_test_path: str = "result/optimization_runs/round19_factorial/splits/internal_test_split.csv",
    max_epochs: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_start_epoch: Optional[int] = None,
) -> Dict:
    jobs = _read_manifest(manifest_path)
    if job_filter:
        jobs = [j for j in jobs if job_filter in str(j.get("job_id", ""))]

    settings_obj = json.loads(Path(settings).read_text(encoding="utf-8"))
    cv = settings_obj.get("screening_cv", {})
    default_epochs = int(max_epochs if max_epochs is not None else cv.get("max_epochs", 500))
    default_patience = int(
        early_stop_patience if early_stop_patience is not None else cv.get("early_stop_patience", 50)
    )
    default_start = int(
        early_stop_start_epoch
        if early_stop_start_epoch is not None
        else cv.get("early_stop_start_epoch", 30)
    )

    pending = []
    for j in jobs:
        status_path = Path(str(j["result_dir"])) / "job_status.json"
        if status_path.is_file():
            prev = json.loads(status_path.read_text(encoding="utf-8"))
            if prev.get("status") == "done":
                continue
        row = dict(j)
        row.setdefault("max_epochs", default_epochs)
        row.setdefault("early_stop_patience", default_patience)
        row.setdefault("early_stop_start_epoch", default_start)
        pending.append(row)
    if limit is not None:
        pending = pending[: int(limit)]

    gpus = _gpu_list()
    pack = int(max_jobs_per_gpu) if max_jobs_per_gpu is not None else 1
    if auto_pack and max_jobs_per_gpu is None:
        pack = recommend_jobs_per_gpu()
    slots = max(1, len(gpus) * max(1, pack))

    # When packing densely, prefer smaller micro-batches first to avoid cascade OOM.
    if pack >= 4:
        default_cands = [256, 128, 64, 32]
    else:
        default_cands = [512, 256, 128, 64, 32]
    candidates = list(micro_batch_candidates or default_cands)

    status_rows: List[dict] = []
    status_path = Path(status_csv or Path(manifest_path).with_name("stage19b_job_status.csv"))

    def _worker(idx_job):
        idx, job = idx_job
        gpu = gpus[idx % len(gpus)] if gpus else ""
        return run_single_job_with_oom_retry(
            pipeline=pipeline,
            job=job,
            micro_batch_candidates=candidates,
            target_effective_batch=target_effective_batch,
            max_retries=max_retries,
            cuda_device=gpu,
            settings=settings,
            response_path=response_path,
            split_assignment=split_assignment,
            internal_test_path=internal_test_path,
        )

    print(
        json.dumps(
            {
                "dispatch_start": True,
                "n_pending": len(pending),
                "gpus": gpus,
                "jobs_per_gpu": pack,
                "gpu_slots": slots,
                "micro_batch_candidates": candidates,
            }
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=slots) as ex:
        futs = {ex.submit(_worker, (i, j)): j for i, j in enumerate(pending)}
        for fut in as_completed(futs):
            row = fut.result()
            status_rows.append(row)
            _write_status_csv(status_path, status_rows)
            print(
                json.dumps(
                    {
                        "job_id": row.get("job_id"),
                        "status": row.get("status"),
                        "elapsed_sec": row.get("elapsed_sec"),
                        "successful_micro_batch": row.get("successful_micro_batch"),
                        "done_so_far": sum(1 for r in status_rows if r.get("status") == "done"),
                        "failed_so_far": sum(1 for r in status_rows if r.get("status") not in {"done"}),
                    }
                ),
                flush=True,
            )

    summary = {
        "manifest": manifest_path,
        "n_pending": len(pending),
        "n_done": sum(1 for r in status_rows if r.get("status") == "done"),
        "n_failed": sum(1 for r in status_rows if r.get("status") not in {"done"}),
        "status_csv": str(status_path),
        "gpu_slots": slots,
        "jobs_per_gpu": pack,
        "gpus": gpus,
        "micro_batch_candidates": candidates,
    }
    Path(status_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 OOM / parallel dispatch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dispatch", help="Dispatch Stage 19B manifest in parallel GPU slots")
    p.add_argument("--manifest", required=True)
    p.add_argument("--pipeline", default="step1_finetune_latent_pipeline_round19.py")
    p.add_argument("--max-jobs-per-gpu", type=int, default=None, help="Override pack count; default=auto")
    p.add_argument("--no-auto-pack", action="store_true")
    p.add_argument("--target-effective-batch", type=int, default=1024)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--job-filter", default=None)
    p.add_argument("--status-csv", default=None)
    p.add_argument("--micro-batch-candidates", default="")
    p.add_argument("--settings", default="config/round19_factorial_settings.json")
    p.add_argument(
        "--response-path",
        default="result/optimization_runs/round19_factorial/data/round19_eligible_response.csv",
    )
    p.add_argument(
        "--split-assignment",
        default="result/optimization_runs/round19_factorial/splits/screening_3fold_assignments.csv",
    )
    p.add_argument(
        "--internal-test-path",
        default="result/optimization_runs/round19_factorial/splits/internal_test_split.csv",
    )

    p_rec = sub.add_parser("recommend-pack", help="Print recommended jobs_per_gpu")
    args = parser.parse_args()

    if args.cmd == "recommend-pack":
        n = recommend_jobs_per_gpu()
        print(json.dumps({"jobs_per_gpu": n}))
        return

    if args.cmd == "dispatch":
        cands = [int(x) for x in str(args.micro_batch_candidates).split(",") if x.strip()]
        out = dispatch_manifest(
            manifest_path=args.manifest,
            pipeline=args.pipeline,
            max_jobs_per_gpu=args.max_jobs_per_gpu,
            auto_pack=not args.no_auto_pack,
            micro_batch_candidates=cands or None,
            target_effective_batch=args.target_effective_batch,
            max_retries=args.max_retries,
            status_csv=args.status_csv,
            limit=args.limit,
            job_filter=args.job_filter,
            settings=args.settings,
            response_path=args.response_path,
            split_assignment=args.split_assignment,
            internal_test_path=args.internal_test_path,
        )
        print(json.dumps(out, indent=2))
        return

    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
