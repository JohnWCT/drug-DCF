#!/usr/bin/env python3
"""Round 23 parallel performance-closure training dispatcher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.training.xa_dispatch import _job_complete, _run_one_job
from concurrent.futures import ProcessPoolExecutor, as_completed
from tools.biocda_telegram_notify import biocda_notify


def build_jobs(config: Dict[str, Any], root: Path, *, include_z: bool) -> List[Dict[str, Any]]:
    models = list(config["models"])
    if include_z and "biocda_xa_z_only" not in models:
        models.append("biocda_xa_z_only")
    jobs = []
    for seed in config["experiment"]["seeds"]:
        for model_type in models:
            jobs.append(
                {
                    "job_id": f"{model_type}_seed{seed}",
                    "model_type": model_type,
                    "seed": int(seed),
                    "run_dir": str(root / f"{model_type}_seed{seed}"),
                }
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/biocda/xa_v2_closure.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-z-ablation", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true", help="Force re-run all jobs")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = ROOT / config["outputs"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)
    max_parallel = int(args.max_parallel or config["training"].get("max_parallel", 3))
    resume = bool(args.resume) and not bool(args.no_resume)

    # Build shared graph cache once before workers (avoids pickle race)
    from biocda.training.graph_cache_io import ensure_graph_cache

    ensure_graph_cache(
        dev_rows_path=ROOT / config["data"]["development_rows"],
        feature_dir=str(ROOT / config["data"]["feature_dir"]),
        drug_smiles_path=str(ROOT / config["data"]["drug_smiles_path"]),
        cache_root=out_root,
        force_rebuild=False,
    )

    jobs = build_jobs(config, out_root, include_z=args.include_z_ablation)
    pending = []
    skipped = []
    for job in jobs:
        if resume and _job_complete(Path(job["run_dir"])):
            skipped.append(job["job_id"])
        else:
            pending.append(job)

    biocda_notify(
        f"Round23 XA closure TRAIN start\npending={len(pending)} skipped={len(skipped)} "
        f"max_parallel={max_parallel} smoke={args.smoke}"
    )

    worker = ROOT / "scripts/train_xa_v2_job.py"
    cmds = [
        [
            sys.executable,
            str(worker),
            "--config",
            str(args.config),
            "--model-type",
            job["model_type"],
            "--seed",
            str(job["seed"]),
            *(["--smoke"] if args.smoke else []),
        ]
        for job in pending
    ]

    results = []
    if pending:
        with ProcessPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(_run_one_job, cmd): cmd for cmd in cmds}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                mt = res["cmd"][res["cmd"].index("--model-type") + 1]
                sd = res["cmd"][res["cmd"].index("--seed") + 1]
                status = "OK" if res["returncode"] == 0 else "FAIL"
                biocda_notify(f"Round23 job {mt}_seed{sd} {status}")
                if res["returncode"] != 0:
                    print(res["stderr_tail"], file=sys.stderr)

    summary = {
        "pending": len(pending),
        "skipped": skipped,
        "results": [
            {"returncode": r["returncode"], "cmd": r["cmd"], "stderr_tail": r.get("stderr_tail", "")[-500:]}
            for r in results
        ],
    }
    (out_root / "dispatch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fails = sum(1 for r in results if r["returncode"] != 0)
    biocda_notify(f"Round23 XA closure TRAIN done fails={fails}/{len(results)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
