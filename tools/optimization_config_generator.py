"""Generate VAEwC pretrain sweep configs and manifest."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_COLUMNS = [
    "job_id",
    "config_path",
    "lambda_proto",
    "proto_temperature",
    "proto_start_epoch",
    "proto_full_epoch",
    "proto_min_samples_per_class",
    "lambda_adv",
    "gan_gen_update_interval",
    "status",
    "result_dir",
    "start_time",
    "end_time",
    "error_message",
]


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _flatten_base_params(base_config: dict) -> dict:
    """Convert pretrain_params grid with singleton lists into one param dict."""
    if "pretrain_param_combinations" in base_config:
        combos = base_config["pretrain_param_combinations"]
        if len(combos) != 1:
            raise ValueError("base_config must contain exactly one pretrain_param_combinations entry")
        return copy.deepcopy(combos[0])
    grid = base_config.get("pretrain_params", {})
    if not grid:
        raise ValueError("base_config must contain pretrain_params or pretrain_param_combinations")
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in grid.items()}


def expand_sweep_combinations(sweep: dict) -> List[dict]:
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def build_generated_config(
    base_params: dict,
    sweep_combo: dict,
    metadata: dict,
) -> dict:
    params = copy.deepcopy(base_params)
    params.update(copy.deepcopy(sweep_combo))
    return {
        "_metadata": metadata,
        "pretrain_param_combinations": [params],
    }


def generate_configs(
    sweep_spec_path: str,
    manifest_dir: str | None = None,
    force: bool = False,
) -> Tuple[str, pd.DataFrame]:
    with open(_resolve_path(sweep_spec_path), "r", encoding="utf-8") as f:
        spec = json.load(f)

    base_config_path = _resolve_path(spec["base_config"])
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)
    base_params = _flatten_base_params(base_config)

    run_id = spec.get("run_id", "vaewc_proto_infonce_round1")
    output_config_dir = _resolve_path(spec.get("output_config_dir", f"config/generated/{run_id}"))
    os.makedirs(output_config_dir, exist_ok=True)

    manifest_root = _resolve_path(manifest_dir or os.path.join("result", "optimization_runs", run_id, "manifests"))
    os.makedirs(manifest_root, exist_ok=True)
    manifest_path = os.path.join(manifest_root, "pretrain_sweep_manifest.csv")

    existing_manifest = None
    if os.path.exists(manifest_path) and not force:
        existing_manifest = pd.read_csv(manifest_path)

    rows = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for idx, combo in enumerate(expand_sweep_combinations(spec["sweep"])):
        job_id = f"exp_proto_{idx:03d}"
        config_name = f"{job_id}.json"
        config_path = os.path.join(output_config_dir, config_name)
        if os.path.exists(config_path) and not force:
            pass
        else:
            metadata = {
                "run_id": run_id,
                "job_id": job_id,
                "sweep_name": run_id,
                "generated_at": generated_at,
                "source_base_config": spec["base_config"],
                **combo,
            }
            payload = build_generated_config(base_params, combo, metadata)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        rel_config_path = os.path.relpath(config_path, PROJECT_ROOT)
        rows.append(
            {
                "job_id": job_id,
                "config_path": rel_config_path,
                "lambda_proto": combo["lambda_proto"],
                "proto_temperature": combo["proto_temperature"],
                "proto_start_epoch": combo["proto_start_epoch"],
                "proto_full_epoch": combo["proto_full_epoch"],
                "proto_min_samples_per_class": combo.get(
                    "proto_min_samples_per_class",
                    combo.get("min_proto_samples_per_class", 1),
                ),
                "lambda_adv": combo["lambda_adv"],
                "gan_gen_update_interval": combo["gan_gen_update_interval"],
                "status": "pending",
                "result_dir": "",
                "start_time": "",
                "end_time": "",
                "error_message": "",
            }
        )

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if existing_manifest is not None and not force:
        merged = existing_manifest.copy()
        for col in MANIFEST_COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
        for _, new_row in manifest_df.iterrows():
            mask = merged["job_id"] == new_row["job_id"]
            if not mask.any():
                merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)
            elif force:
                preserve_cols = ["status", "result_dir", "start_time", "end_time", "error_message"]
                old = merged.loc[mask].iloc[0]
                merged.loc[mask, list(new_row.index)] = new_row.values
                for col in preserve_cols:
                    if col in old and str(old[col]).strip() and old[col] == old[col]:
                        if str(old.get("status", "")) in {"success", "failed", "skipped"}:
                            merged.loc[mask, col] = old[col]
        manifest_df = merged[MANIFEST_COLUMNS]
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path, manifest_df


def main():
    parser = argparse.ArgumentParser("optimization_config_generator")
    parser.add_argument(
        "--sweep-spec",
        default="config/pretrain_sweeps/vaewc_proto_infonce_round1.json",
        help="Sweep specification JSON",
    )
    parser.add_argument("--manifest-dir", default=None, help="Override manifest output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated configs/manifest rows")
    args = parser.parse_args()
    manifest_path, manifest_df = generate_configs(args.sweep_spec, args.manifest_dir, force=args.force)
    print(f"Generated {len(manifest_df)} configs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
