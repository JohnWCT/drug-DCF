#!/usr/bin/env python3
"""CLI: run Stage 20 jobs (single / smoke-pair / run-all)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20_dispatch import dispatch, load_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="result/optimization_runs/round20_unseen_drug_closure/stage20a_dimension/manifest.jsonl",
    )
    p.add_argument("--stage", default="20A", choices=["20A", "20B"])
    p.add_argument("--job-id", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-pair", action="store_true",
                   help="Run C16/C32 (or E3/gated) seed52 fold0 smoke pair")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--max-parallel", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=0, help="unused; smoke uses epochs")
    p.add_argument("--smoke-epochs", type=int, default=1)
    args = p.parse_args()

    manifest = Path(args.manifest)
    if args.stage == "20B" and "stage20a" in str(manifest):
        manifest = Path(
            "result/optimization_runs/round20_unseen_drug_closure/stage20b_predictor/manifest.jsonl"
        )

    job_ids = None
    smoke = bool(args.smoke or args.smoke_pair)
    if args.job_id:
        job_ids = [args.job_id]
    elif args.smoke_pair:
        jobs = load_manifest(manifest)
        # Prefer seed52 fold0 pair
        pair = [
            j["job_id"]
            for j in jobs
            if j["split_seed"] == 52 and j["fold"] == 0
        ]
        # keep first two distinct candidates
        seen = set()
        selected = []
        for jid in pair:
            cand = jid.split("__")[1]
            if cand not in seen:
                seen.add(cand)
                selected.append(jid)
            if len(selected) >= 2:
                break
        job_ids = selected
        print(json.dumps({"smoke_pair": job_ids}))

    if not (args.run_all or job_ids or args.smoke_pair or args.job_id):
        raise SystemExit("Specify --run-all, --job-id, or --smoke-pair")

    summary = dispatch(
        manifest_path=manifest,
        max_parallel=args.max_parallel if args.run_all else min(2, args.max_parallel),
        resume=args.resume and not smoke,
        smoke=smoke,
        smoke_epochs=args.smoke_epochs,
        job_ids=job_ids,
        stage_label=args.stage,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
