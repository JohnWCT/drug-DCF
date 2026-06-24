#!/usr/bin/env python3
"""Build Round 14 VICReg stabilizer pretrain configs and downstream manifests."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_runner import _expand_finetune_combinations
from tools.reconstruction_losses import reconstruction_loss_defaults
from tools.round10_config_builder import ROUND10_DISABLED_LOSSES, _lam_tag
from tools.round12_config_builder import _resolve_round11_baseline
from tools.round13_config_builder import FEATURE_MODE_DEFAULTS
from tools.round9_diagnostics_common import load_json, resolve_path

MANIFEST_COLUMNS = [
    "job_id",
    "exp_id",
    "config_path",
    "result_dir",
    "round",
    "round14_branch",
    "route_id",
    "source_model",
    "lambda_tumor_var",
    "lambda_tumor_cov",
    "tumor_vicreg_start_epoch",
    "tumor_vicreg_full_epoch",
    "conditional_adv_enabled",
    "source_anchor_proto_enabled",
    "lambda_proto_align",
    "reconstruction_loss_type",
    "random_seed",
    "status",
]


def _resolve_pretrain_params(round_root: str, exp_id: str) -> Tuple[dict, str]:
    round_root = resolve_path(round_root)
    pretrain_dir = os.path.join(round_root, "pretrain", exp_id)
    params_path = os.path.join(pretrain_dir, "params.json")
    if os.path.isfile(params_path):
        payload = load_json(params_path)
        params = payload.get("params", payload)
        return copy.deepcopy(params), pretrain_dir

    manifest_path = os.path.join(round_root, "manifests", "pretrain_sweep_manifest.csv")
    if os.path.isfile(manifest_path):
        manifest = pd.read_csv(manifest_path)
        match = manifest[manifest["result_dir"].astype(str).str.endswith(f"/{exp_id}")]
        if not match.empty:
            cfg = resolve_path(str(match.iloc[0]["config_path"]))
            if os.path.isfile(cfg):
                payload = load_json(cfg)
                combos = payload.get("pretrain_param_combinations", [])
                if combos:
                    return copy.deepcopy(combos[0]), pretrain_dir

    raise FileNotFoundError(f"Cannot resolve pretrain params for {exp_id} under {round_root}")


def _write_config(path: str, params: dict, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_metadata": metadata,
        "round14_vicreg_stabilizer": True,
        "pretrain_param_combinations": [params],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _manifest_row(
    job_id: str,
    exp_id: str,
    config_path: str,
    result_dir: str,
    params: dict,
    route_id: str,
    source_model: str,
    branch: str,
) -> dict:
    return {
        "job_id": job_id,
        "exp_id": exp_id,
        "config_path": config_path,
        "result_dir": result_dir,
        "round": params.get("round", "round14"),
        "round14_branch": branch,
        "route_id": route_id,
        "source_model": source_model,
        "lambda_tumor_var": params.get("lambda_tumor_var", 0.0),
        "lambda_tumor_cov": params.get("lambda_tumor_cov", 0.0),
        "tumor_vicreg_start_epoch": params.get("tumor_vicreg_start_epoch", 0),
        "tumor_vicreg_full_epoch": params.get("tumor_vicreg_full_epoch", 0),
        "conditional_adv_enabled": params.get("conditional_adv_enabled", False),
        "source_anchor_proto_enabled": params.get("source_anchor_proto_enabled", False),
        "lambda_proto_align": params.get("lambda_proto_align", 0.0),
        "reconstruction_loss_type": params.get("reconstruction_loss_type", "mse"),
        "random_seed": params.get("random_seed", 0),
        "status": "pending",
    }


def _base_round14_params(baseline_params: dict, route: dict, settings: dict) -> dict:
    params = copy.deepcopy(baseline_params)
    params.update(ROUND10_DISABLED_LOSSES)
    params.update(reconstruction_loss_defaults())
    params.update(
        {
            "round": "round14",
            "route_id": route["route_id"],
            "source_model": route["source_model"],
            "source_route_description": route.get("description", ""),
            "tumor_vicreg_latent_view": "shared",
            "tumor_vicreg_var_target": 1.0,
        }
    )
    params["lambda_tumor_var"] = 0.0
    params["lambda_tumor_cov"] = 0.0
    return params


def _apply_vicreg(
    base: dict,
    branch: str,
    lam_var: float,
    lam_cov: float,
    start: int,
    full: int,
    seed: int,
) -> dict:
    out = copy.deepcopy(base)
    out["round14_branch"] = branch
    out["lambda_tumor_var"] = float(lam_var)
    out["lambda_tumor_cov"] = float(lam_cov)
    out["tumor_vicreg_start_epoch"] = int(start)
    out["tumor_vicreg_full_epoch"] = int(full)
    out["random_seed"] = int(seed)
    return out


def build_round14_pretrain_configs(
    settings_path: str,
    outdir: str,
    force: bool = False,
) -> str:
    settings = load_json(resolve_path(settings_path))
    outdir = resolve_path(outdir)
    config_dir = os.path.join(outdir, "configs")
    manifest_dir = os.path.join(outdir, "manifests")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    manifest_path = os.path.join(manifest_dir, "pretrain_sweep_manifest.csv")
    if os.path.exists(manifest_path) and not force:
        return manifest_path

    round12_root = resolve_path(settings["round12_root"])
    round11_root = resolve_path(settings["round11_root"])
    routes = settings.get("base_routes", [])
    vicreg_cfg = settings.get("vicreg", {})
    schedules = vicreg_cfg.get("schedules", [{"start": 20, "full": 60}, {"start": 40, "full": 90}])
    seeds = [int(s) for s in settings.get("seeds", [101, 202, 303])]

    rows: List[dict] = []
    job_idx = 0
    generated_at = datetime.now(timezone.utc).isoformat()

    def next_exp_id() -> str:
        nonlocal job_idx
        job_idx += 1
        return f"exp_{job_idx:03d}"

    for route in routes:
        route_id = route["route_id"]
        source_model = route["source_model"]
        branch = route.get("round14_branch", "14B" if "008" in source_model else "14C")
        source_round = route.get("source_round", "round12")

        if source_round == "round12":
            baseline_params, checkpoint_dir = _resolve_pretrain_params(round12_root, source_model)
        else:
            baseline_params, checkpoint_dir = _resolve_round11_baseline(round11_root, source_model)

        base = _base_round14_params(baseline_params, route, settings)
        metadata_base = {
            "round": "round14",
            "generated_at": generated_at,
            "route_id": route_id,
            "source_model": source_model,
            "source_checkpoint_dir": checkpoint_dir,
            "settings_path": os.path.relpath(settings_path, PROJECT_ROOT),
        }

        if branch == "14B":
            vicreg_settings = vicreg_cfg.get("settings_main", [])
        else:
            vicreg_settings = vicreg_cfg.get("settings_exp035_small", vicreg_cfg.get("settings_main", []))

        for setting in vicreg_settings:
            lam_var = float(setting.get("lambda_tumor_var", 0.0))
            lam_cov = float(setting.get("lambda_tumor_cov", 0.0))
            for sched in schedules:
                start = int(sched["start"])
                full = int(sched["full"])
                for seed in seeds:
                    params = _apply_vicreg(base, branch, lam_var, lam_cov, start, full, seed)
                    exp_id = next_exp_id()
                    job_id = (
                        f"r14{branch}_{source_model}_vv{_lam_tag(lam_var)}_{_lam_tag(lam_cov)}"
                        f"_s{start}_f{full}_seed{seed}"
                    )
                    config_path = os.path.join(config_dir, f"{job_id}.json")
                    rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                    result_dir = os.path.join(outdir, "pretrain", exp_id)
                    _write_config(
                        config_path,
                        params,
                        {**metadata_base, "branch": branch, "job_id": job_id},
                    )
                    rows.append(
                        _manifest_row(
                            job_id,
                            exp_id,
                            rel_config,
                            os.path.relpath(result_dir, PROJECT_ROOT),
                            params,
                            route_id,
                            source_model,
                            branch,
                        )
                    )

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(manifest_path, index=False)

    meta = {
        "round": "round14",
        "generated_at": generated_at,
        "n_pretrain_jobs": len(rows),
        "n_routes": len(routes),
        "references": settings.get("references", {}),
    }
    with open(os.path.join(outdir, "round14_build_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(rows)} pretrain jobs -> {manifest_path}")
    if "round14_branch" in manifest_df.columns:
        print(manifest_df["round14_branch"].value_counts().to_string())
    return manifest_path


def build_round14_finetune_manifest(
    settings_path: str,
    outdir: str,
    selection_path: str,
    force: bool = False,
) -> Dict[str, str]:
    settings = load_json(resolve_path(settings_path))
    outdir = resolve_path(outdir)
    selection_path = resolve_path(selection_path)
    if not os.path.isfile(selection_path):
        raise FileNotFoundError(f"Selection file not found: {selection_path}")

    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    proto_manifest_path = os.path.join(manifests_dir, "proto_feature_manifest.csv")
    finetune_manifest_path = os.path.join(manifests_dir, "finetune_dispatch_manifest.csv")
    if os.path.exists(finetune_manifest_path) and not force:
        return {
            "proto_feature_manifest": proto_manifest_path,
            "finetune_dispatch_manifest": finetune_manifest_path,
        }

    selection = pd.read_csv(selection_path)
    id_col = "ID" if "ID" in selection.columns else "model_id"
    folder_col = "result_folder" if "result_folder" in selection.columns else "pretrain_result_dir"

    feature_modes = list(settings.get("response_feature_modes", ["none", "own_cancer", "own_plus_summary"]))
    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round14_proto_features.json")
    combos = _expand_finetune_combinations(finetune_config)

    proto_rows: List[dict] = []
    finetune_rows: List[dict] = []

    for _, sel_row in selection.iterrows():
        exp_id = str(sel_row[id_col])
        checkpoint_dir = resolve_path(str(sel_row[folder_col]))
        route_id = str(sel_row.get("route_id", sel_row.get("round14_route_id", "")))
        source_model = str(sel_row.get("source_model", sel_row.get("source_baseline_exp_id", "")))

        for feature_mode in feature_modes:
            defaults = FEATURE_MODE_DEFAULTS.get(feature_mode, FEATURE_MODE_DEFAULTS["own_cancer"])
            job_key = f"r14_{exp_id}_{feature_mode}"
            feature_dir = os.path.join(outdir, "features", exp_id, feature_mode)
            proto_job_id = f"feat_{exp_id}_{feature_mode}"
            proto_rows.append(
                {
                    "job_id": proto_job_id,
                    "source_model_id": exp_id,
                    "source_round": "round14",
                    "checkpoint_dir": checkpoint_dir,
                    "route_id": route_id,
                    "prototype_feature_mode": feature_mode,
                    "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                    "prototype_distance_metric": defaults["prototype_distance_metric"],
                    "include_l2_distance": defaults["include_l2_distance"],
                    "include_same_cancer_gap": defaults["include_same_cancer_gap"],
                    "include_initialized_flag": defaults["include_initialized_flag"],
                    "proto_feature_scaler": defaults["proto_feature_scaler"],
                    "combined_latent_dir": feature_dir,
                    "status": "pending",
                }
            )

            model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
            ms_row = {
                "ID": job_key,
                "model_type": "VAE",
                "result_folder": feature_dir if feature_mode != "none" else checkpoint_dir,
                "selection_rank": int(sel_row.get("selection_rank", 1)),
                "prototype_feature_mode": feature_mode,
                "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                "source_model_id": exp_id,
                "source_round": "round14",
                "route_id": route_id,
                "lineage_source_model": source_model,
            }
            pd.DataFrame([ms_row]).to_csv(model_select_path, index=False)

            for combo in combos:
                ft_job_id = f"ft_{job_key}_c{combo['combo_id']:02d}"
                finetune_rows.append(
                    {
                        "job_id": ft_job_id,
                        "model_id": job_key,
                        "source_model_id": exp_id,
                        "source_round": "round14",
                        "route_id": route_id,
                        "lineage_source_model": source_model,
                        "feature_mode": feature_mode,
                        "prototype_feature_mode": feature_mode,
                        "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                        "combined_latent_dir": feature_dir if feature_mode != "none" else checkpoint_dir,
                        "pretrain_result_dir": checkpoint_dir,
                        "model_select_path": model_select_path,
                        "finetune_config_path": finetune_config,
                        "combo_id": combo["combo_id"],
                        "result_dir": os.path.join(outdir, "finetune", job_key, f"combo_{combo['combo_id']:02d}"),
                        "random_seed": 42,
                        "status": "pending",
                        "start_time": "",
                        "end_time": "",
                        "error_message": "",
                    }
                )

    pd.DataFrame(proto_rows).to_csv(proto_manifest_path, index=False)
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest_path, index=False)

    print(f"Wrote proto feature manifest ({len(proto_rows)} rows) -> {proto_manifest_path}")
    print(f"Wrote finetune manifest ({len(finetune_rows)} jobs) -> {finetune_manifest_path}")
    return {
        "proto_feature_manifest": proto_manifest_path,
        "finetune_dispatch_manifest": finetune_manifest_path,
    }


def build_round14_reference_controls(settings_path: str, outdir: str) -> str:
    settings = load_json(resolve_path(settings_path))
    outdir = resolve_path(outdir)
    reports_dir = os.path.join(outdir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    round13_root = resolve_path(settings["round13_root"])
    agg_path = os.path.join(round13_root, "aggregate", "aggregate_scores.csv")
    controls = settings.get("reference_controls", [])
    rows = []
    if os.path.isfile(agg_path):
        agg = pd.read_csv(agg_path)
        id_col = "Model_ID" if "Model_ID" in agg.columns else "ID"
        for ctrl in controls:
            match = agg[agg[id_col].astype(str) == str(ctrl)]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["control_id"] = ctrl
                rows.append(row)
    out_path = os.path.join(reports_dir, "round14_reference_controls.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote reference controls ({len(rows)} rows) -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 14 VICReg stabilizer configs")
    parser.add_argument("--settings", default="config/round14_vicreg_stabilizer_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round14_vicreg_stabilizer")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build-finetune-manifest", action="store_true")
    parser.add_argument(
        "--selection",
        default="result/optimization_runs/round14_vicreg_stabilizer/selection/pretrain_top10.csv",
    )
    parser.add_argument("--build-reference-controls", action="store_true")
    args = parser.parse_args()

    if args.build_finetune_manifest:
        build_round14_finetune_manifest(args.settings, args.outdir, args.selection, force=args.force)
    elif args.build_reference_controls:
        build_round14_reference_controls(args.settings, args.outdir)
    else:
        build_round14_pretrain_configs(args.settings, args.outdir, force=args.force)


if __name__ == "__main__":
    main()
