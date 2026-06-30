#!/usr/bin/env python3
"""Build Round 16 focused brute-force downstream optimization manifests."""

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
from tools.prototype_response_features import (
    get_projected_context_dim,
    get_projected_delta_dim,
    is_own_proto_context_mode,
    is_own_proto_delta_replacement_mode,
    resolve_feature_mode_options,
)
from tools.round13_config_builder import FEATURE_MODE_DEFAULTS
from tools.round9_diagnostics_common import load_json, resolve_path
from tools.round14_config_builder import (
    _apply_vicreg,
    _manifest_row,
    _resolve_pretrain_params,
    _write_config,
)
from tools.round10_config_builder import _lam_tag

ROUND16_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "r13_exp_008": {
        "source_model_id": "exp_008",
        "source_round": "round12",
        "checkpoint_root_key": "round12_root",
        "checkpoint_subdir": "exp_008",
    },
    "r13_exp_035": {
        "source_model_id": "exp_035",
        "source_round": "round11",
        "checkpoint_root_key": "round11_root",
        "checkpoint_subdir": "exp_035",
    },
    "r15c_exp_005": {
        "source_model_id": "exp_005",
        "source_round": "round15",
        "checkpoint_root_key": "round15_root",
        "checkpoint_subdir": "exp_005",
    },
    "r15c_exp_024": {
        "source_model_id": "exp_024",
        "source_round": "round15",
        "checkpoint_root_key": "round15_root",
        "checkpoint_subdir": "exp_024",
    },
}


def _resolve_checkpoint(settings: dict, model_key: str) -> Tuple[str, dict]:
    spec = ROUND16_MODEL_SPECS[model_key]
    root = resolve_path(settings[spec["checkpoint_root_key"]])
    checkpoint_dir = os.path.join(root, "pretrain", spec["checkpoint_subdir"])
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found for {model_key}: {checkpoint_dir}")
    return checkpoint_dir, spec


def _feature_defaults(feature_mode: str) -> dict:
    if feature_mode == "none":
        return FEATURE_MODE_DEFAULTS["none"]
    if is_own_proto_context_mode(feature_mode) or is_own_proto_delta_replacement_mode(feature_mode) or feature_mode == "own_plus_summary":
        base = FEATURE_MODE_DEFAULTS["own_plus_summary"]
    else:
        base = FEATURE_MODE_DEFAULTS.get("own_plus_summary", {})
    opts = resolve_feature_mode_options(
        feature_mode,
        include_l2_distance=base["include_l2_distance"],
        include_same_cancer_gap=base["include_same_cancer_gap"],
        include_initialized_flag=base["include_initialized_flag"],
        proto_feature_scaler=base["proto_feature_scaler"],
    )
    return {
        "prototype_distance_metric": "cosine",
        "include_l2_distance": opts["include_l2_distance"],
        "include_same_cancer_gap": opts["include_same_cancer_gap"],
        "include_initialized_flag": opts["include_initialized_flag"],
        "proto_feature_scaler": opts["proto_feature_scaler"],
    }


def _load_combos(settings: dict, max_combos: Optional[int] = None) -> List[dict]:
    finetune_cfg = settings.get("finetune", {}).get("config", "config/params_finetune_round16_bruteforce.json")
    combos = _expand_finetune_combinations(finetune_cfg)
    if max_combos is not None:
        combos = [c for c in combos if int(c["combo_id"]) < int(max_combos)]
    return combos


def _build_model_pool(settings: dict, model_keys: List[str]) -> pd.DataFrame:
    rows = []
    for model_key in model_keys:
        checkpoint_dir, spec = _resolve_checkpoint(settings, model_key)
        rows.append(
            {
                "model_id": model_key,
                "source_model_id": spec["source_model_id"],
                "source_round": spec["source_round"],
                "checkpoint_dir": checkpoint_dir,
            }
        )
    return pd.DataFrame(rows)


def _own_proto_manifest_fields(feature_mode: str) -> dict:
    mode = str(feature_mode).lower()
    proj_dim = get_projected_delta_dim(mode) or get_projected_context_dim(mode)
    uses_projection = mode in (
        "own_proto_context_projected_16",
        "own_proto_context_projected_32",
        "own_proto_delta_projected_16",
        "own_proto_delta_projected_32",
    )
    flags = {
        "uses_own_plus_summary": mode
        in ("own_plus_summary", "own_plus_summary_no_delta_control", "own_plus_summary_plus_delta"),
        "uses_delta": mode
        in (
            "own_proto_delta",
            "own_proto_delta_only",
            "own_plus_summary_plus_delta",
            "own_proto_delta_normed",
            "own_proto_delta_projected_16",
            "own_proto_delta_projected_32",
        ),
        "uses_projection": uses_projection,
    }
    return {
        **flags,
        "requires_projection": uses_projection,
        "projection_dim": int(proj_dim),
        "projection_fit_domain": "source_only" if uses_projection else "",
    }


def _append_finetune_jobs(
    *,
    settings: dict,
    outdir: str,
    stage: str,
    model_pool: pd.DataFrame,
    feature_modes: List[str],
    combos: List[dict],
    seeds: List[int],
    finetune_rows: List[dict],
    proto_rows: List[dict],
    model_select_dir: str,
    combo_filter: Optional[set] = None,
    feature_root: Optional[str] = None,
    finetune_root: Optional[str] = None,
) -> None:
    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round16_bruteforce.json")
    features_base = feature_root or os.path.join(outdir, "features")
    finetune_base = finetune_root or os.path.join(outdir, "finetune", stage)

    for _, model in model_pool.iterrows():
        model_key = str(model["model_id"])
        source_id = str(model["source_model_id"])
        checkpoint_dir = str(model["checkpoint_dir"])
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
            job_key = f"{model_key}_{feature_mode}"
            feature_dir = os.path.join(features_base, model_key, feature_mode)
            proto_job_id = f"feat_{model_key}_{feature_mode}"

            proto_rows.append(
                {
                    "job_id": proto_job_id,
                    "stage": stage,
                    "model_id": model_key,
                    "source_model_id": source_id,
                    "source_round": model["source_round"],
                    "checkpoint_dir": checkpoint_dir,
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
                    "status": "pending",
                    **_own_proto_manifest_fields(feature_mode),
                }
            )

            model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
            ms_row = {
                "ID": job_key,
                "model_type": "VAE",
                "result_folder": checkpoint_dir if feature_mode == "none" else feature_dir,
                "selection_rank": 1,
                "prototype_feature_mode": opts["feature_mode_label"],
                "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                "source_model_id": source_id,
                "source_round": model["source_round"],
                "round16_model_key": model_key,
            }
            pd.DataFrame([ms_row]).to_csv(model_select_path, index=False)

            for combo in combos:
                combo_id = int(combo["combo_id"])
                if combo_filter is not None and combo_id not in combo_filter:
                    continue
                for seed in seeds:
                    ft_job_id = f"ft_{stage}_{job_key}_c{combo_id:02d}_s{seed}"
                    result_dir = os.path.join(
                        finetune_base,
                        job_key,
                        f"combo_{combo_id:02d}",
                        f"seed_{seed}",
                    )
                    finetune_rows.append(
                        {
                            "job_id": ft_job_id,
                            "stage": stage,
                            "model_id": job_key,
                            "source_model_id": source_id,
                            "source_round": model["source_round"],
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
                            "batch_size": combo.get("batch_size", settings.get("finetune", {}).get("batch_size", 12288)),
                            "mini_batch_size": combo.get(
                                "mini_batch_size", settings.get("finetune", {}).get("mini_batch_size", 3072)
                            ),
                            "epochs": combo.get("epochs", settings.get("finetune", {}).get("epochs", 1500)),
                            "result_dir": result_dir,
                            "status": "pending",
                            "start_time": "",
                            "end_time": "",
                            "error_message": "",
                        }
                    )


def build_stage16a(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings["stage16a"]
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    model_pool = _build_model_pool(settings, list(stage_cfg["models"]))
    combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 24))
    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []

    _append_finetune_jobs(
        settings=settings,
        outdir=outdir,
        stage="16a",
        model_pool=model_pool,
        feature_modes=list(stage_cfg["feature_modes"]),
        combos=combos,
        seeds=list(stage_cfg["seeds"]),
        finetune_rows=finetune_rows,
        proto_rows=proto_rows,
        model_select_dir=model_select_dir,
    )

    pool_path = os.path.join(manifests_dir, "model_pool.csv")
    model_pool.to_csv(pool_path, index=False)
    proto_manifest = os.path.join(manifests_dir, "proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)
    stage_manifest = os.path.join(manifests_dir, "stage16a_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(stage_manifest, index=False)

    return {
        "model_pool": pool_path,
        "proto_feature_manifest": proto_manifest,
        "finetune_dispatch_manifest": finetune_manifest,
        "stage16a_manifest": stage_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage16b(settings: dict, outdir: str, top_candidates_path: str) -> Dict[str, str]:
    candidates = pd.read_csv(resolve_path(top_candidates_path))
    if candidates.empty:
        raise ValueError(f"No candidates found in {top_candidates_path}")

    stage_cfg = settings["stage16b"]
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects_stage16b")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []
    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round16_bruteforce.json")
    all_combos = _load_combos(settings)

    for _, cand in candidates.iterrows():
        model_key = str(cand.get("round16_model_key", cand.get("model_key", "")))
        if not model_key or model_key not in ROUND16_MODEL_SPECS:
            model_id = str(cand.get("model_id", ""))
            for key in ROUND16_MODEL_SPECS:
                if key in model_id:
                    model_key = key
                    break
        feature_mode = str(cand.get("feature_mode", cand.get("prototype_feature_mode", "none")))
        combo_id = int(cand["combo_id"])
        checkpoint_dir, spec = _resolve_checkpoint(settings, model_key)
        job_key = f"{model_key}_{feature_mode}"
        feature_dir = os.path.join(outdir, "features", model_key, feature_mode)

        base = FEATURE_MODE_DEFAULTS["own_plus_summary"]
        opts = resolve_feature_mode_options(
            feature_mode,
            include_l2_distance=base["include_l2_distance"],
            include_same_cancer_gap=base["include_same_cancer_gap"],
            include_initialized_flag=base["include_initialized_flag"],
            proto_feature_scaler=base["proto_feature_scaler"],
        )
        proto_rows.append(
            {
                "job_id": f"feat_{job_key}",
                "stage": "16b",
                "model_id": model_key,
                "source_model_id": spec["source_model_id"],
                "source_round": spec["source_round"],
                "checkpoint_dir": checkpoint_dir,
                "prototype_feature_mode": opts["feature_mode_label"],
                "feature_mode": feature_mode,
                "combined_latent_dir": feature_dir,
                "status": "pending",
            }
        )

        src_ms = os.path.join(outdir, "manifests", "model_selects", f"{job_key}.csv")
        model_select_path = src_ms if os.path.isfile(src_ms) else os.path.join(model_select_dir, f"{job_key}.csv")
        if not os.path.isfile(model_select_path):
            ms_row = {
                "ID": job_key,
                "model_type": "VAE",
                "result_folder": checkpoint_dir if feature_mode == "none" else feature_dir,
                "selection_rank": 1,
                "prototype_feature_mode": opts["feature_mode_label"],
                "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                "source_model_id": spec["source_model_id"],
                "source_round": spec["source_round"],
                "round16_model_key": model_key,
            }
            pd.DataFrame([ms_row]).to_csv(model_select_path, index=False)

        combo_row = all_combos[combo_id]
        for seed in stage_cfg["seeds"]:
            ft_job_id = f"ft_16b_{job_key}_c{combo_id:02d}_s{seed}"
            finetune_rows.append(
                {
                    "job_id": ft_job_id,
                    "stage": "16b",
                    "model_id": job_key,
                    "source_model_id": spec["source_model_id"],
                    "source_round": spec["source_round"],
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
                    "batch_size": combo_row.get("batch_size", 12288),
                    "mini_batch_size": combo_row.get("mini_batch_size", 3072),
                    "epochs": combo_row.get("epochs", 1500),
                    "result_dir": os.path.join(
                        outdir,
                        "stage16b",
                        "finetune",
                        job_key,
                        f"combo_{combo_id:02d}",
                        f"seed_{seed}",
                    ),
                    "status": "pending",
                    "start_time": "",
                    "end_time": "",
                    "error_message": "",
                }
            )

    proto_manifest = os.path.join(manifests_dir, "stage16b_proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "stage16b_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)
    confirm_manifest = os.path.join(manifests_dir, "stage16b_confirmation_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(confirm_manifest, index=False)

    return {
        "stage16b_proto_feature_manifest": proto_manifest,
        "stage16b_finetune_dispatch_manifest": finetune_manifest,
        "stage16b_confirmation_manifest": confirm_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage16c(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings["stage16c"]
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects_stage16c")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    model_pool = _build_model_pool(settings, list(stage_cfg["models"]))
    combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 8))
    combo_ids = {int(c["combo_id"]) for c in combos}
    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []

    _append_finetune_jobs(
        settings=settings,
        outdir=outdir,
        stage="16c",
        model_pool=model_pool,
        feature_modes=list(stage_cfg["feature_variants"]),
        combos=combos,
        seeds=list(stage_cfg["seeds"]),
        finetune_rows=finetune_rows,
        proto_rows=proto_rows,
        model_select_dir=model_select_dir,
        combo_filter=combo_ids,
    )

    proto_manifest = os.path.join(manifests_dir, "stage16c_proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "stage16c_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)

    return {
        "stage16c_proto_feature_manifest": proto_manifest,
        "stage16c_finetune_dispatch_manifest": finetune_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage16e(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings.get("stage16e_own_proto_context", settings.get("stage16e", {}))
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects_stage16e")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    model_pool = _build_model_pool(settings, list(stage_cfg["models"]))
    combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 8))
    combo_ids = {int(c["combo_id"]) for c in combos}
    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []

    _append_finetune_jobs(
        settings=settings,
        outdir=outdir,
        stage="16e",
        model_pool=model_pool,
        feature_modes=list(stage_cfg["feature_modes"]),
        combos=combos,
        seeds=list(stage_cfg["seeds"]),
        finetune_rows=finetune_rows,
        proto_rows=proto_rows,
        model_select_dir=model_select_dir,
        combo_filter=combo_ids,
        feature_root=os.path.join(outdir, "features_stage16e"),
        finetune_root=os.path.join(outdir, "stage16e", "finetune"),
    )

    proto_manifest = os.path.join(manifests_dir, "stage16e_proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "stage16e_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)

    return {
        "stage16e_proto_feature_manifest": proto_manifest,
        "stage16e_finetune_dispatch_manifest": finetune_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage16f(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings.get("stage16f_delta_replacement", settings.get("stage16f", {}))
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects_stage16f")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    feature_modes = list(stage_cfg["feature_modes"])
    if stage_cfg.get("include_optional_modes"):
        for mode in stage_cfg.get("optional_feature_modes", []):
            if mode not in feature_modes:
                feature_modes.append(mode)

    finetune_cfg = stage_cfg.get(
        "finetune_config",
        settings.get("finetune", {}).get("config", "config/params_finetune_round16_delta_replacement.json"),
    )
    stage_settings = dict(settings)
    stage_settings["finetune"] = {**settings.get("finetune", {}), "config": finetune_cfg}

    model_pool = _build_model_pool(settings, list(stage_cfg["models"]))
    combos = _load_combos(stage_settings, max_combos=stage_cfg.get("max_combos", 8))
    combo_ids = {int(c["combo_id"]) for c in combos}
    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []

    _append_finetune_jobs(
        settings=stage_settings,
        outdir=outdir,
        stage="16f",
        model_pool=model_pool,
        feature_modes=feature_modes,
        combos=combos,
        seeds=list(stage_cfg["seeds"]),
        finetune_rows=finetune_rows,
        proto_rows=proto_rows,
        model_select_dir=model_select_dir,
        combo_filter=combo_ids,
        feature_root=os.path.join(outdir, "features_stage16f"),
        finetune_root=os.path.join(outdir, "stage16f", "finetune"),
    )

    proto_manifest = os.path.join(manifests_dir, "stage16f_proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "stage16f_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)

    return {
        "stage16f_proto_feature_manifest": proto_manifest,
        "stage16f_finetune_dispatch_manifest": finetune_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage16d(settings: dict, outdir: str) -> Dict[str, str]:
    """Ultra-low / late VICReg micro-search on Round 15 lineages (pretrain only)."""
    stage_cfg = settings.get("stage16d", {})
    if not stage_cfg.get("enabled", False):
        raise ValueError("stage16d is disabled in settings")

    stage_root = os.path.join(resolve_path(outdir), "stage16d")
    config_dir = os.path.join(stage_root, "configs")
    manifests_dir = os.path.join(stage_root, "manifests")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    round15_root = resolve_path(settings["round15_root"])
    lineages = list(stage_cfg.get("lineages", []))
    vicreg_lambdas = [float(x) for x in stage_cfg.get("vicreg_lambdas", [0.0])]
    schedules = list(stage_cfg.get("schedules", [{"start": 90, "full": 150}]))
    seeds = [int(s) for s in stage_cfg.get("seeds", [101, 202, 303])]

    generated_at = datetime.now(timezone.utc).isoformat()
    metadata_base = {
        "round": "round16",
        "stage": "16d",
        "generated_at": generated_at,
        "purpose": "vicreg_micro_search",
    }

    rows: List[dict] = []
    job_idx = 0

    def next_exp_id() -> str:
        nonlocal job_idx
        job_idx += 1
        return f"exp_{job_idx:03d}"

    for lineage_key in lineages:
        if lineage_key not in ROUND16_MODEL_SPECS:
            raise ValueError(f"Unknown 16D lineage model key: {lineage_key}")
        spec = ROUND16_MODEL_SPECS[lineage_key]
        source_model = spec["source_model_id"]
        baseline_params, checkpoint_dir = _resolve_pretrain_params(round15_root, source_model)
        base = copy.deepcopy(baseline_params)
        base.update(
            {
                "round": "round16",
                "round16_branch": "16D",
                "round16_lineage": lineage_key,
                "source_model": source_model,
                "lineage_checkpoint_dir": checkpoint_dir,
            }
        )

        for lam in vicreg_lambdas:
            for sched in schedules:
                start = int(sched["start"])
                full = int(sched["full"])
                for seed in seeds:
                    params = _apply_vicreg(base, "16D", lam, lam, start, full, seed)
                    params["round"] = "round16"
                    params["round16_branch"] = "16D"
                    params["round16_lineage"] = lineage_key
                    exp_id = next_exp_id()
                    job_id = (
                        f"r16d_{lineage_key}_vv{_lam_tag(lam)}_{_lam_tag(lam)}"
                        f"_s{start}_f{full}_seed{seed}"
                    )
                    config_path = os.path.join(config_dir, f"{job_id}.json")
                    rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                    result_dir = os.path.join(stage_root, "pretrain", exp_id)
                    _write_config(
                        config_path,
                        params,
                        {
                            **metadata_base,
                            "lineage": lineage_key,
                            "source_model": source_model,
                            "job_id": job_id,
                        },
                    )
                    row = _manifest_row(
                        job_id,
                        exp_id,
                        rel_config,
                        os.path.relpath(result_dir, PROJECT_ROOT),
                        params,
                        route_id=lineage_key,
                        source_model=source_model,
                        branch="16D",
                    )
                    row["round"] = "round16"
                    row["round16_branch"] = "16D"
                    row["round16_lineage"] = lineage_key
                    rows.append(row)

    manifest_path = os.path.join(manifests_dir, "stage16d_pretrain_manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return {
        "stage16d_pretrain_manifest": manifest_path,
        "stage16d_root": stage_root,
        "n_jobs": len(rows),
    }


def build_round16_configs(
    settings_path: str,
    outdir: str,
    stage: str,
    top_candidates: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    del force
    settings = load_json(settings_path)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    stage = stage.lower().replace("stage", "").strip()
    if stage == "16a":
        outputs = build_stage16a(settings, outdir)
    elif stage == "16b":
        if not top_candidates:
            raise ValueError("--top-candidates is required for stage 16b")
        outputs = build_stage16b(settings, outdir, top_candidates)
    elif stage == "16c":
        outputs = build_stage16c(settings, outdir)
    elif stage == "16e":
        outputs = build_stage16e(settings, outdir)
    elif stage == "16f":
        outputs = build_stage16f(settings, outdir)
    elif stage == "16d":
        outputs = build_stage16d(settings, outdir)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    meta = {
        "round": "round16",
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
        "references": settings.get("references", {}),
    }
    meta_path = os.path.join(outdir, f"round16_build_{stage}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Round 16 stage {stage}: wrote {outputs.get('n_jobs', '?')} finetune jobs -> {outdir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 16 brute-force manifests")
    parser.add_argument("--settings", default="config/round16_bruteforce_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round16_bruteforce")
    parser.add_argument("--stage", required=True, choices=["16a", "16b", "16c", "16d", "16e", "16f"])
    parser.add_argument("--top-candidates", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_round16_configs(args.settings, args.outdir, args.stage, args.top_candidates, force=args.force)


if __name__ == "__main__":
    main()
