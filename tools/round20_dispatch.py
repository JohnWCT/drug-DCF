#!/usr/bin/env python3
"""Round 20 GPU job dispatcher with resume, OOM retry, and Telegram notify.

Designed to keep ~90% of system throughput by packing many light-GPU GIN jobs
onto the same RTX 6000 Ada. Individual jobs use ~20 MB VRAM; the bottleneck is
CPU graph collation, so high concurrency (default 16) is preferred over huge
micro-batches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.telegram_notify import send_telegram_message  # noqa: E402

OOM_EXIT = 42


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _notify(msg: str) -> None:
    try:
        send_telegram_message(msg, fail_silently=True)
    except Exception:  # noqa: BLE001
        pass


def _job_complete(job_dir: Path) -> bool:
    status = job_dir / "status.json"
    metrics = job_dir / "metrics.json"
    ckpt = job_dir / "best_checkpoint.pt"
    if not (status.is_file() and metrics.is_file() and ckpt.is_file()):
        return False
    try:
        st = json.loads(status.read_text(encoding="utf-8"))
        mt = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return st.get("status") == "COMPLETE" and mt.get("status") == "COMPLETE" and not mt.get("smoke")


def _build_cmd(job: dict, *, micro_batch: int, accum: int, smoke: bool, smoke_epochs: int) -> List[str]:
    cmd = [
        sys.executable,
        "step1_finetune_latent_pipeline_round20_cv.py",
        "--job-id", job["job_id"],
        "--candidate-id", job["candidate_id"],
        "--context-id", job["context_id"],
        "--predictor-kind", job.get("predictor_kind", "pooled_e3"),
        "--feature-dir", job["feature_store_path"],
        "--expected-omics-dim", str(job["omics_dim"]),
        "--split-assignment", job["split_assignment_path"],
        "--split-seed", str(job["split_seed"]),
        "--fold-id", str(job["fold"]),
        "--model-seed", str(job["model_seed"]),
        "--result-dir", job["output_dir"],
        "--micro-batch-size", str(micro_batch),
        "--accumulation-steps", str(accum),
        "--e3-contract",
        "result/optimization_runs/round20_unseen_drug_closure/stage20_0/resolved_e3.json",
        "--response-path", job.get(
            "response_path",
            "result/optimization_runs/round19_factorial/splits/development_rows.csv",
        ),
    ]
    if smoke:
        cmd += ["--smoke", "--smoke-epochs", str(smoke_epochs)]
    return cmd


def _run_one(job: dict, *, smoke: bool, smoke_epochs: int, micro_batch: int, accum: int) -> dict:
    """Worker entrypoint (must be top-level for ProcessPoolExecutor)."""
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    job_dir = Path(job["output_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"

    cur_micro, cur_accum = micro_batch, accum
    # Keep effective batch = 1024 if possible: when OOM, halve micro and double accum.
    for attempt in range(4):
        cmd = _build_cmd(job, micro_batch=cur_micro, accum=cur_accum, smoke=smoke, smoke_epochs=smoke_epochs)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {_utc()} attempt={attempt} micro={cur_micro} accum={cur_accum} =====\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        if proc.returncode == 0:
            return {"job_id": job["job_id"], "status": "COMPLETE", "attempt": attempt,
                    "micro_batch": cur_micro, "accum": cur_accum}
        if proc.returncode == OOM_EXIT:
            if cur_micro <= 32:
                return {"job_id": job["job_id"], "status": "FAILED", "error": "OOM_exhausted"}
            cur_micro = max(32, cur_micro // 2)
            cur_accum = max(1, (micro_batch * accum) // cur_micro)
            continue
        return {"job_id": job["job_id"], "status": "FAILED", "returncode": proc.returncode}
    return {"job_id": job["job_id"], "status": "FAILED", "error": "retries_exhausted"}


def load_manifest(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dispatch(
    *,
    manifest_path: Path,
    max_parallel: int = 16,
    resume: bool = True,
    smoke: bool = False,
    smoke_epochs: int = 1,
    job_ids: Optional[List[str]] = None,
    micro_batch: int = 256,
    accum: int = 4,
    stage_label: str = "20A",
) -> dict:
    jobs = load_manifest(manifest_path)
    if job_ids:
        wanted = set(job_ids)
        jobs = [j for j in jobs if j["job_id"] in wanted]
    pending = []
    skipped = []
    for job in jobs:
        if job.get("skip_train"):
            skipped.append(job["job_id"])
            continue
        if resume and not smoke and _job_complete(Path(job["output_dir"])):
            skipped.append(job["job_id"])
        else:
            pending.append(job)

    _notify(
        f"[Round20 Stage {stage_label}] start\n"
        f"total={len(jobs)} pending={len(pending)} skipped_complete={len(skipped)}\n"
        f"max_parallel={max_parallel} smoke={smoke}"
    )
    print(json.dumps({"pending": len(pending), "skipped": len(skipped), "max_parallel": max_parallel}))

    results = []
    t0 = time.time()
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futs = {
            pool.submit(_run_one, job, smoke=smoke, smoke_epochs=smoke_epochs,
                        micro_batch=micro_batch, accum=accum): job
            for job in pending
        }
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"job_id": job["job_id"], "status": "FAILED", "error": str(exc)}
            results.append(res)
            done += 1
            if res.get("status") != "COMPLETE":
                failed += 1
            if done % 5 == 0 or done == len(pending):
                elapsed = (time.time() - t0) / 3600
                _notify(
                    f"[Round20 Stage {stage_label}] progress {done}/{len(pending)} "
                    f"failed={failed} elapsed_h={elapsed:.2f}"
                )
            print(json.dumps(res), flush=True)

    summary = {
        "stage": stage_label,
        "total": len(jobs),
        "pending": len(pending),
        "skipped_complete": len(skipped),
        "completed_now": sum(1 for r in results if r.get("status") == "COMPLETE"),
        "failed": failed,
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": results,
        "finished_at": _utc(),
    }
    out = manifest_path.parent / "dispatch_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _notify(
        f"[Round20 Stage {stage_label}] DONE\n"
        f"completed_now={summary['completed_now']} failed={failed} "
        f"skipped={len(skipped)} elapsed_h={summary['elapsed_seconds']/3600:.2f}"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--max-parallel", type=int, default=16)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-epochs", type=int, default=1)
    p.add_argument("--job-id", action="append", default=None)
    p.add_argument("--micro-batch-size", type=int, default=256)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--stage-label", default="20A")
    args = p.parse_args()
    summary = dispatch(
        manifest_path=Path(args.manifest),
        max_parallel=args.max_parallel,
        resume=not args.no_resume,
        smoke=args.smoke,
        smoke_epochs=args.smoke_epochs,
        job_ids=args.job_id,
        micro_batch=args.micro_batch_size,
        accum=args.accumulation_steps,
        stage_label=args.stage_label,
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
