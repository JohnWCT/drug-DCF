#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.round20_stage20b import build_stage20b_manifest
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dimension-lock", required=True)
    p.add_argument("--outdir", default="result/optimization_runs/round20_unseen_drug_closure/stage20b_predictor")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    report = build_stage20b_manifest(dimension_lock_path=Path(args.dimension_lock), outdir=Path(args.outdir))
    print(json.dumps(report, indent=2))
    if args.dry_run:
        print("DRY_RUN_OK")
if __name__ == "__main__":
    main()
