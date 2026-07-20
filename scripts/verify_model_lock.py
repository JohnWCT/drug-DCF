#!/usr/bin/env python3
"""Verify biocda_final_model_lock.json consistency."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.utils.hashing import sha256_file


def verify_lock(path: Path, *, strict: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if payload.get("tcga_used_for_selection"):
        issues.append("tcga_used_for_selection_must_be_false")
    for ckpt, digest in zip(payload.get("checkpoint_paths", []), payload.get("checkpoint_hashes", [])):
        p = Path(ckpt)
        if not p.is_file():
            issues.append(f"missing_checkpoint:{ckpt}")
            continue
        if sha256_file(p) != digest:
            issues.append(f"hash_mismatch:{ckpt}")
    status = payload.get("status", "UNKNOWN")
    if strict and status != "LOCKED":
        issues.append(f"status_not_locked:{status}")
    ok = not issues
    return {"ok": ok, "status": status, "issues": issues}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-manifest",
        type=Path,
        default=ROOT / "reports/biocda_final_model_lock.json",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = verify_lock(args.lock_manifest, strict=args.strict)
    print(json.dumps(report, indent=2))
    if not report["ok"] and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
