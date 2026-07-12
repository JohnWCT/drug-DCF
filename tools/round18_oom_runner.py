"""Round 18 OOM-safe micro-batch probing and process-level job dispatch."""
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


OOM_EXIT_CODE = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class OOMProbeResult:
    requested_micro_batch: int
    successful_micro_batch: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    oom_retry_count: int
    oom_batch_history: List[int] = field(default_factory=list)
    amp_enabled: bool = True
    status: str = "ok"
    detail: str = ""


def compute_accumulation_steps(target_effective_batch: int, micro_batch: int) -> int:
    if micro_batch <= 0:
        raise ValueError("micro_batch must be > 0")
    return int(math.ceil(target_effective_batch / micro_batch))


def probe_micro_batch(
    candidates: Sequence[int],
    *,
    target_effective_batch: int = 1024,
    max_retries: int = 4,
    amp_enabled: bool = True,
    try_fn: Optional[Callable[[int], None]] = None,
) -> OOMProbeResult:
    """
    Try micro-batch sizes in order. try_fn(batch) should raise RuntimeError('CUDA out of memory')
    or a custom exception with 'out of memory' in message to simulate OOM.
    """
    history: List[int] = []
    attempts = 0
    last_requested = int(candidates[0]) if candidates else 0

    for batch in candidates:
        if attempts > max_retries:
            break
        last_requested = int(batch)
        attempts += 1
        if try_fn is None:
            accum = compute_accumulation_steps(target_effective_batch, batch)
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=int(batch),
                gradient_accumulation_steps=accum,
                effective_batch_size=int(batch * accum),
                oom_retry_count=0,
                oom_batch_history=[],
                amp_enabled=amp_enabled,
                status="ok",
                detail="dry_probe_no_try_fn",
            )
        try:
            try_fn(int(batch))
            accum = compute_accumulation_steps(target_effective_batch, batch)
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=int(batch),
                gradient_accumulation_steps=accum,
                effective_batch_size=int(batch * accum),
                oom_retry_count=len(history),
                oom_batch_history=list(history),
                amp_enabled=amp_enabled,
                status="ok",
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                history.append(int(batch))
                continue
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=-1,
                gradient_accumulation_steps=-1,
                effective_batch_size=-1,
                oom_retry_count=len(history),
                oom_batch_history=list(history),
                amp_enabled=amp_enabled,
                status="hard_failure",
                detail=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=-1,
                gradient_accumulation_steps=-1,
                effective_batch_size=-1,
                oom_retry_count=len(history),
                oom_batch_history=list(history),
                amp_enabled=amp_enabled,
                status="hard_failure",
                detail=str(exc),
            )

    return OOMProbeResult(
        requested_micro_batch=last_requested,
        successful_micro_batch=-1,
        gradient_accumulation_steps=-1,
        effective_batch_size=-1,
        oom_retry_count=len(history),
        oom_batch_history=list(history),
        amp_enabled=amp_enabled,
        status="oom_exhausted",
        detail="all micro-batch candidates failed",
    )


def write_resource_metadata(outdir: str, probe: OOMProbeResult, extra: Optional[dict] = None) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = asdict(probe)
    if extra:
        payload.update(extra)
    (out / "runtime_resource_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history = {
        "oom_retry_count": probe.oom_retry_count,
        "oom_batch_history": probe.oom_batch_history,
        "status": probe.status,
    }
    (out / "oom_retry_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return payload


def detect_gpu_job_slots(jobs_per_gpu: int = 1) -> int:
    try:
        import torch

        n = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        n = 0
    if n <= 0:
        return 1
    return max(1, n * max(1, jobs_per_gpu))


def _halve_batch(batch: int) -> int:
    return max(1, int(batch) // 2)


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


def run_single_job_with_oom_retry(
    *,
    pipeline: str,
    job: dict,
    micro_batch_candidates: Sequence[int],
    target_effective_batch: int,
    max_retries: int,
    cuda_device: str,
    python_exe: str = sys.executable,
) -> dict:
    """
    Process-level OOM retry:
      exit 42 -> halve micro-batch -> fresh subprocess from epoch 0.
    """
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
    }

    for attempt, micro_batch in enumerate(candidates):
        if attempt > max_retries:
            break
        accum = compute_accumulation_steps(target_effective_batch, int(micro_batch))
        cmd = [
            python_exe,
            pipeline,
            "--mode",
            "train_fold",
            "--architecture-family",
            str(job["architecture_family"]),
            "--omics-mode",
            str(job["omics_mode"]),
            "--feature-dir",
            str(job["feature_dir"]),
            "--split-assignment",
            str(job["split_assignment"]),
            "--fold-id",
            str(job["fold_id"]),
            "--response-path",
            str(job["response_data_path"]),
            "--drug-smiles-path",
            str(job["drug_smiles_path"]),
            "--result-dir",
            str(result_dir),
            "--micro-batch-size",
            str(int(micro_batch)),
            "--accumulation-steps",
            str(accum),
            "--model-seed",
            str(job.get("model_seed", 101)),
        ]
        if job.get("transformer_config_id"):
            cmd.extend(["--transformer-config-id", str(job["transformer_config_id"])])
        if job.get("residual_mode"):
            cmd.extend(["--residual-mode", str(job["residual_mode"])])
        if job.get("global_lr"):
            cmd.extend(["--global-lr", str(job["global_lr"])])
        if job.get("max_epochs"):
            cmd.extend(["--max-epochs", str(job["max_epochs"])])
        if job.get("max_batches"):
            cmd.extend(["--max-batches", str(job["max_batches"])])

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
            status["oom_retry_count"] = len(history)
            status["oom_batch_history"] = list(history)
            (result_dir / "job_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            return status

        if proc.returncode == OOM_EXIT_CODE:
            history.append(int(micro_batch))
            status["status"] = "oom_retry"
            status["oom_retry_count"] = len(history)
            status["oom_batch_history"] = list(history)
            # Clear any partial checkpoint before next clean process
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
    pipeline: str = "step1_finetune_latent_pipeline_round18_cv.py",
    max_jobs_per_gpu: int = 1,
    micro_batch_candidates: Optional[Sequence[int]] = None,
    target_effective_batch: int = 1024,
    max_retries: int = 4,
    status_csv: Optional[str] = None,
    limit: Optional[int] = None,
    job_filter: Optional[str] = None,
) -> Dict:
    jobs = _read_manifest(manifest_path)
    if job_filter:
        jobs = [j for j in jobs if job_filter in str(j.get("job_id", ""))]
    # skip already done
    pending = []
    for j in jobs:
        status_path = Path(str(j["result_dir"])) / "job_status.json"
        if status_path.is_file():
            prev = json.loads(status_path.read_text(encoding="utf-8"))
            if prev.get("status") == "done":
                continue
        pending.append(j)
    if limit is not None:
        pending = pending[: int(limit)]

    candidates = list(micro_batch_candidates or [512, 256, 128, 64, 32])
    gpus = _gpu_list()
    slots = max(1, len(gpus) * max(1, max_jobs_per_gpu))
    status_rows: List[dict] = []
    status_path = Path(status_csv or Path(manifest_path).with_name("job_status.csv"))

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
        )

    with ThreadPoolExecutor(max_workers=slots) as ex:
        futs = {ex.submit(_worker, (i, j)): j for i, j in enumerate(pending)}
        for fut in as_completed(futs):
            status_rows.append(fut.result())
            _write_status_csv(status_path, status_rows)

    summary = {
        "manifest": manifest_path,
        "n_pending": len(pending),
        "n_done": sum(1 for r in status_rows if r.get("status") == "done"),
        "n_failed": sum(1 for r in status_rows if r.get("status") not in {"done"}),
        "status_csv": str(status_path),
        "gpu_slots": slots,
    }
    Path(status_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 18 OOM runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dispatch = sub.add_parser("dispatch", help="Dispatch manifest jobs with process-level OOM retry")
    p_dispatch.add_argument("--manifest", required=True)
    p_dispatch.add_argument("--pipeline", default="step1_finetune_latent_pipeline_round18_cv.py")
    p_dispatch.add_argument("--max-jobs-per-gpu", type=int, default=1)
    p_dispatch.add_argument("--target-effective-batch", type=int, default=1024)
    p_dispatch.add_argument("--max-retries", type=int, default=4)
    p_dispatch.add_argument("--limit", type=int, default=None)
    p_dispatch.add_argument("--job-filter", default=None)
    p_dispatch.add_argument("--status-csv", default=None)
    p_dispatch.add_argument(
        "--micro-batch-candidates",
        default="512,256,128,64,32",
        help="Comma-separated micro-batch sizes",
    )

    p_run = sub.add_parser("run-job", help="Run one manifest row with OOM retry")
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--job-id", required=True)
    p_run.add_argument("--pipeline", default="step1_finetune_latent_pipeline_round18_cv.py")
    p_run.add_argument("--cuda-device", default="0")
    p_run.add_argument("--target-effective-batch", type=int, default=1024)
    p_run.add_argument("--max-retries", type=int, default=4)
    p_run.add_argument(
        "--micro-batch-candidates",
        default="512,256,128,64,32",
    )

    args = parser.parse_args()
    if args.cmd == "dispatch":
        cands = [int(x) for x in str(args.micro_batch_candidates).split(",") if x.strip()]
        out = dispatch_manifest(
            manifest_path=args.manifest,
            pipeline=args.pipeline,
            max_jobs_per_gpu=args.max_jobs_per_gpu,
            micro_batch_candidates=cands,
            target_effective_batch=args.target_effective_batch,
            max_retries=args.max_retries,
            status_csv=args.status_csv,
            limit=args.limit,
            job_filter=args.job_filter,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "run-job":
        jobs = {j["job_id"]: j for j in _read_manifest(args.manifest)}
        if args.job_id not in jobs:
            raise SystemExit(f"job_id not found: {args.job_id}")
        cands = [int(x) for x in str(args.micro_batch_candidates).split(",") if x.strip()]
        out = run_single_job_with_oom_retry(
            pipeline=args.pipeline,
            job=jobs[args.job_id],
            micro_batch_candidates=cands,
            target_effective_batch=args.target_effective_batch,
            max_retries=args.max_retries,
            cuda_device=str(args.cuda_device),
        )
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
