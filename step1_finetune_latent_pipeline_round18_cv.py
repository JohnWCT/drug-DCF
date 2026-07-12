#!/usr/bin/env python3
"""Round 18 CV finetune pipeline entrypoint (smoke + staged training)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_oom_runner import detect_gpu_job_slots, probe_micro_batch, write_resource_metadata
from tools.round18_train_loop import run_synthetic_smoke_train


def run_smoke(args: argparse.Namespace) -> dict:
    families = [
        ("pooled_mlp", "pure"),
        ("pooled_transformer", "pure"),
        ("cross_attention", "pure"),
        ("cross_attention", "pooled_residual"),
    ]
    results = []
    for family, residual in families:
        out = run_synthetic_smoke_train(family, residual_mode=residual, steps=args.steps)
        results.append(out)
        print(json.dumps({"smoke": family, "residual": residual, "train_loss": out["train"]["loss"]}, indent=2))

    probe = probe_micro_batch(
        [512, 256, 128, 64, 32],
        target_effective_batch=1024,
        try_fn=None,
    )
    meta_dir = Path(args.outdir) / "pipeline_smoke_resources"
    write_resource_metadata(
        str(meta_dir),
        probe,
        extra={
            "gpu_slots": detect_gpu_job_slots(1),
            "smoke_architectures": [r["architecture_family"] for r in results],
        },
    )
    summary = {
        "ok": True,
        "n_architectures": len(results),
        "results": results,
        "resource_dir": str(meta_dir),
    }
    out_path = Path(args.outdir) / "pipeline_smoke_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "n_architectures": len(results)}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 18 CV pipeline")
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
