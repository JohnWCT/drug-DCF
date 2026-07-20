#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.round20_release import audit_release
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release-dir", required=True)
    p.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()
    audit_release(release_dir=Path(args.release_dir), strict=args.strict)
if __name__ == "__main__":
    main()
