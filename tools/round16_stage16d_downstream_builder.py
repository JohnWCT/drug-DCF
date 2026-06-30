#!/usr/bin/env python3
"""Build Stage 16D downstream feature + finetune manifests from filtered pretrain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_runner import _expand_finetune_combinations
from tools.round16_bruteforce_config_builder import ROUND16_MODEL_SPECS, _feature_defaults, _own_proto_manifest_fields
from tools.round9_diagnostics_common import load_json, resolve_path
from tools.prototype_response_features import resolve_feature_mode_options
from tools.round13_config_builder import FEATURE_MODE_DEFAULTS


def build_stage16d_downstream_manifests(
    settings_path: str,
    candidates_path: str,
    stage_root: str,
    *,
    force: bool = False,
) -> Dict[str, str]:
    settings = load_json(resolve_path(settings_path))
    stage_root = resolve_path(stage_root)
    candidates_path = resolve_path(candidates_path)
    downstream_cfg = settings.get("stage16d", {}).get("downstream", {})
    feature_modes = list(downstream_cfg.get("feature_modes", ["none", "own_plus_summary"]))
    seeds = [int(s) for s in downstream_cfg.get("seeds", [101, 202, 303])]
    max_combos = int(downstream_cfg.get("max_combos", 1))

    manifests_dir = os.path.join(stage_root, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects_stage16d_downstream")
    features_root = os.path.join(stage_root, "features_downstream")
    finetune_root = os.path.join(stage_root, "downstream", "finetune")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    proto_manifest = os.path.join(manifests_dir, "stage16d_downstream_proto_feature_manifest.csv")
    finetune_manifest = os.path.join(manifests_dir, "stage16d_downstream_finetune_manifest.csv")
    if os.path.isfile(finetune_manifest) and not force:
        return {
            "proto_feature_manifest": proto_manifest,
            "finetune_dispatch_manifest": finetune_manifest,
        }

    candidates = pd.read_csv(candidates_path)
    if candidates.empty:
        raise ValueError(f"No candidates in {candidates_path}")

    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round16_bruteforce.json")
    combos = _expand_finetune_combinations(finetune_config)
    combos = [c for c in combos if int(c["combo_id"]) < max_combos]

    proto_rows: List[dict] = []
    finetune_rows: List[dict] = []

    for _, cand in candidates.iterrows():
        lineage = str(cand["round16_lineage"])
        spec = ROUND16_MODEL_SPECS[lineage]
        source_id = str(spec["source_model_id"])
        source_round = str(spec["source_round"])
        checkpoint_dir = resolve_path(str(cand["pretrain_result_dir"]))
        downstream_model_id = str(cand["downstream_model_id"])
        exp_id = str(cand["exp_id"])

        for feature_mode in feature_modes:
            defaults = _feature_defaults(feature_mode)
            base = FEATURE_MODE_DEFAULTS["own_plus_summary"]
            opts = resolve_feature_mode_options(
                feature_mode,
                include_l2_distance=base["include_l2_distance"],
                include_same_cancer_gap=base["include_same_cancer_gap"],
                include_initialized_flag=base["include_initialized_flag"],
                proto_feature_scaler=base["proto_feature_scaler"],
            )
            job_key = f"{downstream_model_id}_{feature_mode}"
            feature_dir = os.path.join(features_root, downstream_model_id, feature_mode)
            proto_job_id = f"feat_16d_{downstream_model_id}_{feature_mode}"

            proto_rows.append(
                {
                    "job_id": proto_job_id,
                    "stage": "16d",
                    "model_id": downstream_model_id,
                    "exp_id": exp_id,
                    "source_model_id": source_id,
                    "source_round": source_round,
                    "round16_lineage": lineage,
                    "checkpoint_dir": os.path.relpath(checkpoint_dir, PROJECT_ROOT),
                    "pretrain_result_dir": os.path.relpath(checkpoint_dir, PROJECT_ROOT),
                    "prototype_feature_mode": opts["feature_mode_label"],
                    "feature_mode": feature_mode,
                    "feature_variant": feature_mode,
                    "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                    "prototype_distance_metric": defaults["prototype_distance_metric"],
                    "include_l2_distance": defaults["include_l2_distance"],
                    "include_same_cancer_gap": defaults["include_same_cancer_gap"],
                    "include_initialized_flag": defaults["include_initialized_flag"],
                    "proto_feature_scaler": defaults["proto_feature_scaler"],
                    "combined_latent_dir": feature_dir,
                    "lambda_tumor_var": cand.get("lambda_tumor_var"),
                    "lambda_tumor_cov": cand.get("lambda_tumor_cov"),
                    "tumor_vicreg_start_epoch": cand.get("tumor_vicreg_start_epoch"),
                    "selection_reason": cand.get("selection_reason", ""),
                    "status": "pending",
                    **_own_proto_manifest_fields(feature_mode),
                }
            )

            model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
            ms_row = {
                "ID": job_key,
                "model_type": "VAE",
                "result_folder": feature_dir if feature_mode != "none" else checkpoint_dir,
                "selection_rank": 1,
                "prototype_feature_mode": opts["feature_mode_label"],
                "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                "source_model_id": source_id,
                "source_round": source_round,
                "round16_model_key": lineage,
                "round16_branch": "16D",
                "round16_pretrain_exp": exp_id,
            }
            pd.DataFrame([ms_row]).to_csv(model_select_path, index=False)

            for combo in combos:
                combo_id = int(combo["combo_id"])
                for seed in seeds:
                    ft_job_id = f"ft_16d_{job_key}_c{combo_id:02d}_s{seed}"
                    result_dir = os.path.join(
                        finetune_root,
                        downstream_model_id,
                        feature_mode,
                        f"combo_{combo_id:02d}",
                        f"seed_{seed}",
                    )
                    finetune_rows.append(
                        {
                            "job_id": ft_job_id,
                            "stage": "16d",
                            "model_id": job_key,
                            "downstream_model_id": downstream_model_id,
                            "exp_id": exp_id,
                            "source_model_id": source_id,
                            "source_round": source_round,
                            "round16_lineage": lineage,
                            "feature_mode": feature_mode,
                            "feature_variant": feature_mode,
                            "prototype_feature_mode": opts["feature_mode_label"],
                            "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                            "combined_latent_dir": feature_dir if feature_mode != "none" else checkpoint_dir,
                            "pretrain_result_dir": checkpoint_dir,
                            "model_select_path": model_select_path,
                            "finetune_config_path": finetune_config,
                            "combo_id": combo_id,
                            "seed": seed,
                            "batch_size": settings.get("finetune", {}).get("batch_size", 24576),
                            "mini_batch_size": settings.get("finetune", {}).get("mini_batch_size", 6144),
                            "epochs": settings.get("finetune", {}).get("epochs", 1500),
                            "lambda_tumor_var": cand.get("lambda_tumor_var"),
                            "lambda_tumor_cov": cand.get("lambda_tumor_cov"),
                            "tumor_vicreg_start_epoch": cand.get("tumor_vicreg_start_epoch"),
                            "selection_reason": cand.get("selection_reason", ""),
                            "result_dir": result_dir,
                            "status": "pending",
                            "start_time": "",
                            "end_time": "",
                            "error_message": "",
                        }
                    )

    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)
    meta = {
        "n_candidates": len(candidates),
        "n_proto_jobs": len(proto_rows),
        "n_finetune_jobs": len(finetune_rows),
        "feature_modes": feature_modes,
        "seeds": seeds,
        "max_combos": max_combos,
    }
    meta_path = os.path.join(stage_root, "reports", "stage16d_downstream_build_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(proto_rows)} proto jobs, {len(finetune_rows)} finetune jobs")
    return {
        "proto_feature_manifest": proto_manifest,
        "finetune_dispatch_manifest": finetune_manifest,
        "n_finetune_jobs": len(finetune_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 16D downstream manifests")
    parser.add_argument("--settings", default="config/round16_bruteforce_settings.json")
    parser.add_argument(
        "--candidates",
        default="result/optimization_runs/round16_bruteforce/stage16d/reports/stage16d_pretrain_candidates.csv",
    )
    parser.add_argument("--stage-root", default="result/optimization_runs/round16_bruteforce/stage16d")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_stage16d_downstream_manifests(
        args.settings,
        args.candidates,
        args.stage_root,
        force=args.force,
    )


if __name__ == "__main__":
    main()
