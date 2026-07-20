#!/usr/bin/env python3
"""CLI: build Stage 20A splits (if needed) + 30-job paired manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20_stage20a import build_stage20a_manifest, build_stage20a_splits  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/round20/stage20a_dimension.yaml")
    p.add_argument(
        "--e3-contract",
        default="result/optimization_runs/round20_unseen_drug_closure/stage20_0/resolved_e3.json",
    )
    p.add_argument(
        "--output",
        default="result/optimization_runs/round20_unseen_drug_closure/stage20a_dimension/manifest.jsonl",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rebuild-splits", action="store_true")
    args = p.parse_args()

    split_dir = ROOT / "result/optimization_runs/round20_unseen_drug_closure/splits"
    if args.rebuild_splits or not (split_dir / "round20a_drug_heldout_seed52_assignments.csv").is_file():
        audit = build_stage20a_splits(outdir=split_dir)
        print(json.dumps({"splits": audit["status"], "n_folds": len(audit["folds"])}))

    report = build_stage20a_manifest(
        resolved_e3_path=Path(args.e3_contract),
        outdir=Path(args.output).parent,
    )
    print(json.dumps(report, indent=2))
    if report["missing_pairs"] or report["jobs_total"] != 30:
        raise SystemExit("manifest validation failed")
    if args.dry_run:
        print("DRY_RUN_OK")


if __name__ == "__main__":
    main()
