#!/usr/bin/env python3
"""Build Round 9 3-seed reproduction configs and pretrain manifest."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from typing import List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import load_json, resolve_path


def _read_baseline_params(checkpoint_dir: str) -> dict:
    params_path = os.path.join(resolve_path(checkpoint_dir), "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing params.json for reproduction: {params_path}")
    payload = load_json(params_path)
    params = copy.deepcopy(payload.get("params", payload))
    if not params:
        raise ValueError(f"Empty params in {params_path}")
    return params


def build_reproduction_manifest(
    resolved_baselines_path: str,
    baseline_config_path: str,
    outdir: str,
    force: bool = False,
) -> str:
    outdir = resolve_path(outdir)
    config_dir = os.path.join(outdir, "configs")
    manifest_dir = os.path.join(outdir, "manifests")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    resolved = pd.read_csv(resolve_path(resolved_baselines_path))
    spec = load_json(baseline_config_path)
    seeds: List[int] = [int(s) for s in spec.get("seeds", [101, 202, 303])]

    manifest_path = os.path.join(manifest_dir, "pretrain_sweep_manifest.csv")
    if os.path.exists(manifest_path) and not force:
        return manifest_path

    rows = []
    generated_at = datetime.now(timezone.utc).isoformat()
    job_idx = 0
    for _, baseline in resolved[resolved["resolved"] == True].iterrows():  # noqa: E712
        exp_id = str(baseline["exp_id"])
        role = str(baseline["role"])
        checkpoint_dir = str(baseline["checkpoint_dir"])
        params = _read_baseline_params(checkpoint_dir)
        for seed in seeds:
            job_id = f"r9_{exp_id}_seed{seed}"
            config_name = f"{exp_id}_seed{seed}.json"
            config_path = os.path.join(config_dir, config_name)
            repro_params = copy.deepcopy(params)
            repro_params["random_seed"] = int(seed)
            payload = {
                "_metadata": {
                    "round": "round9",
                    "generated_at": generated_at,
                    "source_exp_id": exp_id,
                    "source_role": role,
                    "source_checkpoint_dir": checkpoint_dir,
                },
                "round9_reproduction": True,
                "source_exp_id": exp_id,
                "source_role": role,
                "reproduction_seed": int(seed),
                "diagnostics_enabled": True,
                "save_latent_for_round9_diagnostics": True,
                "save_deconfounding_qc_metrics": True,
                "pretrain_param_combinations": [repro_params],
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            rows.append(
                {
                    "job_id": job_id,
                    "config_path": os.path.relpath(config_path, PROJECT_ROOT),
                    "source_exp_id": exp_id,
                    "source_role": role,
                    "reproduction_seed": seed,
                    "status": "pending",
                    "result_dir": "",
                    "start_time": "",
                    "end_time": "",
                    "error_message": "",
                }
            )
            job_idx += 1

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 9 reproduction manifest")
    parser.add_argument("--resolved-baselines", required=True)
    parser.add_argument("--baseline-config", default="config/round9_baselines.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round9_reproduction")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_reproduction_manifest(
        args.resolved_baselines,
        args.baseline_config,
        args.outdir,
        force=args.force,
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
