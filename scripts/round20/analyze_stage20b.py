#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.round20_stage20b import analyze_stage20b
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="result/optimization_runs/round20_unseen_drug_closure/stage20b_predictor")
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--no-strict", action="store_true")
    args = p.parse_args()
    print(json.dumps(analyze_stage20b(stage_dir=Path(args.input_dir), strict=not args.no_strict), indent=2))
if __name__ == "__main__":
    main()
