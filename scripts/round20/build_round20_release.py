#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.round20_release import build_release
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-lock", required=True)
    p.add_argument("--tcga-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    build_release(model_lock_path=Path(args.model_lock), tcga_dir=Path(args.tcga_dir), output_dir=Path(args.output_dir))
    print("RELEASE_BUILD_OK")
if __name__ == "__main__":
    main()
