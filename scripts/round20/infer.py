#!/usr/bin/env python3
"""Round 20 inference entry (preflight + delegate to release pipeline)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.result_contracts import load_json, sha256_file


def preflight(*, release_dir: Path, strict: bool = True) -> dict:
    release_dir = Path(release_dir)
    lock_path = release_dir / "configs/final_model_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing lock: {lock_path}")
    lock = load_json(lock_path)
    n_ckpt = len(list((release_dir / "checkpoints").glob("*.pt")))
    info = {
        "model_lock_sha256": sha256_file(lock_path),
        "selected_context": lock["selected_context"]["id"],
        "expected_omics_dimension": lock["selected_context"]["omics_dimension"],
        "checkpoint_count": n_ckpt,
        "expected_checkpoint_count": 15,
    }
    print("=== Round 20 inference preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if strict and n_ckpt != 15:
        raise SystemExit("Preflight failed: checkpoint count mismatch")
    return info


def run_infer(
    *,
    release_dir: Path,
    mode: str,
    response_file: Path,
    output: Path,
    omics_file: Path | None = None,
    strict: bool = True,
) -> None:
    preflight(release_dir=release_dir, strict=strict)
    if mode not in {"frozen_latent", "raw_omics"}:
        raise ValueError(mode)
    if mode == "raw_omics" and omics_file is None:
        raise SystemExit("raw_omics mode requires --omics-file")
    # Full GPU inference requires release checkpoints on device; document path for operators.
    raise SystemExit(
        "Inference runner requires CUDA + release checkpoints. "
        "Use scripts/round20/run_round20_tcga.py pattern with --release-dir for batch scoring. "
        f"Preflight OK for mode={mode}, response={response_file}, output={output}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--release-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["frozen_latent", "raw_omics"], required=True)
    p.add_argument("--response-file", type=Path, required=True)
    p.add_argument("--omics-file", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    run_infer(
        release_dir=args.release_dir,
        mode=args.mode,
        response_file=args.response_file,
        omics_file=args.omics_file,
        output=args.output,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
