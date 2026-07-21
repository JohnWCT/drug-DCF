"""Parallel GPU dispatch for BioCDA XA validation training."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.biocda_telegram_notify import biocda_notify


def _job_complete(run_dir: Path) -> bool:
    return (run_dir / "metrics_by_seed.json").is_file() and (run_dir / "best.pt").is_file()


def _run_one_job(cmd: List[str]) -> Dict[str, Any]:
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }


def build_job_list(config: Dict[str, Any], root: Path) -> List[Dict[str, Any]]:
    jobs = []
    for seed in config["experiment"]["seeds"]:
        for model_type in config["models"]:
            run_id = f"{model_type}_seed{seed}"
            jobs.append(
                {
                    "job_id": run_id,
                    "model_type": model_type,
                    "seed": int(seed),
                    "run_dir": str(root / run_id),
                }
            )
    return jobs


def dispatch_training(
    *,
    config_path: Path,
    config: Dict[str, Any],
    max_parallel: int = 4,
    resume: bool = True,
) -> Dict[str, Any]:
    out_root = ROOT / config["outputs"]["root"]
    jobs = build_job_list(config, out_root)
    pending = []
    skipped = []
    for job in jobs:
        if resume and _job_complete(Path(job["run_dir"])):
            skipped.append(job["job_id"])
        else:
            pending.append(job)

    biocda_notify(
        f"Round21 parallel TRAIN start\npending={len(pending)} skipped={len(skipped)} "
        f"max_parallel={max_parallel}"
    )

    worker_script = ROOT / "scripts/run_xa_train_job.py"
    results: List[Dict[str, Any]] = []
    if not pending:
        return {"status": "COMPLETE", "pending": 0, "skipped": skipped, "results": results}

    cmds = [
        [
            sys.executable,
            str(worker_script),
            "--config",
            str(config_path),
            "--model-type",
            job["model_type"],
            "--seed",
            str(job["seed"]),
        ]
        for job in pending
    ]

    with ProcessPoolExecutor(max_workers=int(max_parallel)) as pool:
        futures = {pool.submit(_run_one_job, cmd): cmd for cmd in cmds}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            job_id = res["cmd"][-3] + "_seed" + res["cmd"][-1]
            status = "OK" if res["returncode"] == 0 else "FAIL"
            biocda_notify(f"Round21 job {job_id} {status}")

    failed = [r for r in results if r["returncode"] != 0]
    summary = {
        "status": "COMPLETE" if not failed else "PARTIAL_FAIL",
        "pending": len(pending),
        "skipped": skipped,
        "failed": len(failed),
        "results": results,
    }
    (out_root / "parallel_dispatch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    biocda_notify(
        f"Round21 parallel TRAIN done status={summary['status']} failed={len(failed)}"
    )
    if failed:
        raise RuntimeError(f"{len(failed)} training jobs failed; see parallel_dispatch_summary.json")
    return summary
