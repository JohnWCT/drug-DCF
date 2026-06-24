#!/usr/bin/env python3
"""Build Round 15 reproducibility + exp_008 route rescue configs and manifests."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_runner import _expand_finetune_combinations
from tools.reconstruction_losses import reconstruction_loss_defaults
from tools.round10_config_builder import ROUND10_DISABLED_LOSSES, _lam_tag
from tools.round12_config_builder import _resolve_round11_baseline
from tools.round13_config_builder import FEATURE_MODE_DEFAULTS
from tools.round14_config_builder import (
    MANIFEST_COLUMNS as PRETRAIN_MANIFEST_COLUMNS,
    _apply_vicreg,
    _base_round14_params,
    _manifest_row,
    _resolve_pretrain_params,
    _write_config,
)
from tools.round9_diagnostics_common import load_json, resolve_path

COMPACT_FEATURE_MODES = ("none", "own_plus_summary")

PRETRAIN_MANIFEST_COLUMNS_ROUND15 = list(PRETRAIN_MANIFEST_COLUMNS)
if "round15_branch" not in PRETRAIN_MANIFEST_COLUMNS_ROUND15:
    idx = PRETRAIN_MANIFEST_COLUMNS_ROUND15.index("round14_branch") + 1
    PRETRAIN_MANIFEST_COLUMNS_ROUND15.insert(idx, "round15_branch")


def _compact_defaults(feature_mode: str) -> dict:
    return FEATURE_MODE_DEFAULTS.get(feature_mode, FEATURE_MODE_DEFAULTS["own_plus_summary"])


def _resolve_checkpoint(round_root: str, model_id: str) -> str:
    path = os.path.join(resolve_path(round_root), "pretrain", model_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _load_round14_pretrain_rows(round14_root: str) -> pd.DataFrame:
    rows: List[dict] = []
    round14_root = resolve_path(round14_root)
    pretrain_dir = os.path.join(round14_root, "pretrain")
    manifest_path = os.path.join(round14_root, "manifests", "pretrain_sweep_manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.isfile(manifest_path) else pd.DataFrame()
    manifest_by_exp: Dict[str, dict] = {}
    if not manifest.empty and "exp_id" in manifest.columns:
        for _, row in manifest.iterrows():
            manifest_by_exp[str(row["exp_id"])] = row.to_dict()

    for summary_path in sorted(glob.glob(os.path.join(pretrain_dir, "exp_*", "run_summary.json"))):
        exp_dir = os.path.dirname(summary_path)
        exp_id = os.path.basename(exp_dir)
        payload = load_json(summary_path)
        params = payload.get("params", {})
        metrics = payload.get("metrics", {})
        manifest_row = manifest_by_exp.get(exp_id, {})
        route_id = str(params.get("route_id", manifest_row.get("route_id", "")))
        branch = str(params.get("round14_branch", manifest_row.get("round14_branch", "")))
        source_model = str(params.get("source_model", manifest_row.get("source_model", "")))
        lam_var = float(params.get("lambda_tumor_var", manifest_row.get("lambda_tumor_var", 0)) or 0)
        lam_cov = float(params.get("lambda_tumor_cov", manifest_row.get("lambda_tumor_cov", 0)) or 0)
        rows.append(
            {
                "model_id": exp_id,
                "checkpoint_dir": exp_dir,
                "route_id": route_id,
                "round14_branch": branch,
                "source_model": source_model,
                "lambda_tumor_var": lam_var,
                "lambda_tumor_cov": lam_cov,
                "lambda_sum": lam_var + lam_cov,
                "tumor_vicreg_start_epoch": int(
                    params.get("tumor_vicreg_start_epoch", manifest_row.get("tumor_vicreg_start_epoch", 0)) or 0
                ),
                "random_seed": int(params.get("random_seed", manifest_row.get("random_seed", 0)) or 0),
                "kmeans_ari": metrics.get("kmeans_ari", params.get("kmeans_ari")),
                "status": payload.get("status", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _pick_round14_14b_exp008_candidates(round14_root: str) -> List[Dict[str, Any]]:
    df = _load_round14_pretrain_rows(round14_root)
    if df.empty:
        return []

    mask = (
        df["round14_branch"].astype(str).eq("14B")
        | df["route_id"].astype(str).str.contains("exp008", case=False, na=False)
        | df["source_model"].astype(str).str.contains("008", na=False)
    )
    sub = df[mask].copy()
    if sub.empty:
        return []

    picks: List[Dict[str, Any]] = []
    seen_roles: Set[str] = set()

    def _append(role: str, row: pd.Series) -> None:
        if role in seen_roles:
            return
        seen_roles.add(role)
        picks.append(
            {
                "pool_model_id": str(row["model_id"]),
                "role": role,
                "source_round": "round14",
                "checkpoint_dir": str(row["checkpoint_dir"]),
                "route_id": str(row.get("route_id", "exp008_proto_response_route")),
                "lineage_source_model": str(row.get("source_model", "exp_008")),
                "round14_branch": "14B",
            }
        )

    no_vicreg = sub[sub["lambda_sum"].fillna(0) <= 0]
    if not no_vicreg.empty:
        best = no_vicreg.sort_values("kmeans_ari", ascending=False, na_position="last").iloc[0]
        _append("round14_14b_exp008_no_vicreg", best)

    pos = sub[sub["lambda_sum"].fillna(0) > 0].copy()
    if not pos.empty:
        low = pos.sort_values("lambda_sum", ascending=True).iloc[0]
        _append("round14_14b_exp008_lowest_vicreg", low)
        late = pos.sort_values("tumor_vicreg_start_epoch", ascending=False).iloc[0]
        _append("round14_14b_exp008_late_vicreg", late)

    return picks


def resolve_round15b_model_pool(settings: dict) -> Tuple[List[Dict[str, Any]], List[str]]:
    round12_root = resolve_path(settings["round12_root"])
    round11_root = resolve_path(settings.get("round11_root", "result/optimization_runs/round11_stability_recon"))
    round14_root = resolve_path(settings["round14_root"])
    pool_cfg = settings.get("round15b_forced_exp008_route", {})
    max_models = int(pool_cfg.get("max_models", 6))
    warnings_out: List[str] = []
    selected: List[Dict[str, Any]] = []
    seen_checkpoints: Set[str] = set()

    def _add(entry: Dict[str, Any]) -> None:
        ckpt = str(entry["checkpoint_dir"])
        if ckpt in seen_checkpoints:
            return
        seen_checkpoints.add(ckpt)
        selected.append(entry)

    source_model = str(pool_cfg.get("source_model", settings.get("round15a_repro", {}).get("source_model", "exp_008")))
    try:
        _add(
            {
                "pool_model_id": source_model,
                "role": "round13_best_source_exp008",
                "source_round": "round12",
                "checkpoint_dir": _resolve_checkpoint(round12_root, source_model),
                "route_id": "exp008_proto_response_route",
                "lineage_source_model": source_model,
                "round15_branch": "15B",
            }
        )
    except FileNotFoundError as exc:
        warnings_out.append(str(exc))

    if pool_cfg.get("include_round14_14b_candidates", True):
        for cand in _pick_round14_14b_exp008_candidates(round14_root):
            cand["round15_branch"] = "15B"
            _add(cand)

    if pool_cfg.get("include_round13_exp035_zonly", True):
        try:
            _add(
                {
                    "pool_model_id": "exp_035",
                    "role": "round13_exp035_zonly_reference",
                    "source_round": "round11",
                    "checkpoint_dir": _resolve_checkpoint(round11_root, "exp_035"),
                    "route_id": "exp035_strong_zonly_route",
                    "lineage_source_model": "exp_035",
                    "round15_branch": "15B",
                }
            )
        except FileNotFoundError:
            warnings_out.append("Round 11 exp_035 checkpoint missing; skipped")

    if pool_cfg.get("include_round14_best_exp078", True):
        try:
            _add(
                {
                    "pool_model_id": "exp_078",
                    "role": "round14_best_exp078_reference",
                    "source_round": "round14",
                    "checkpoint_dir": _resolve_checkpoint(round14_root, "exp_078"),
                    "route_id": "exp035_strong_zonly_route",
                    "lineage_source_model": "exp_035",
                    "round14_branch": "14C",
                    "round15_branch": "15B",
                }
            )
        except FileNotFoundError:
            warnings_out.append("Round 14 exp_078 checkpoint missing; skipped")

    if len(selected) < 2:
        warnings_out.append("Round 15B model pool smaller than expected")

    return selected[:max_models], warnings_out


def _append_finetune_block(
    finetune_rows: List[dict],
    proto_rows: List[dict],
    model_select_dir: str,
    outdir: str,
    model: Dict[str, Any],
    feature_modes: List[str],
    finetune_config: str,
    combos: List[dict],
    branch: str,
    random_seed: int = 42,
    seed_suffix: str = "",
) -> None:
    pool_id = str(model.get("pool_model_id", model.get("model_id", model.get("source_model_id", "unknown"))))
    checkpoint_dir = str(model["checkpoint_dir"])
    source_round = str(model.get("source_round", "round15"))
    route_id = str(model.get("route_id", ""))
    lineage = str(model.get("lineage_source_model", pool_id))

    for feature_mode in feature_modes:
        if feature_mode not in COMPACT_FEATURE_MODES:
            continue
        defaults = _compact_defaults(feature_mode)
        suffix = f"_{seed_suffix}" if seed_suffix else ""
        job_key = f"r15{branch.lower()}_{pool_id}_{feature_mode}{suffix}"
        feature_dir = os.path.join(outdir, "features", branch, pool_id, feature_mode)
        if seed_suffix:
            feature_dir = os.path.join(feature_dir, seed_suffix)

        proto_job_id = f"feat_{branch}_{pool_id}_{feature_mode}{suffix}"
        proto_rows.append(
            {
                "job_id": proto_job_id,
                "source_model_id": pool_id,
                "source_round": source_round,
                "checkpoint_dir": checkpoint_dir,
                "role": model.get("role", ""),
                "round15_branch": branch,
                "route_id": route_id,
                "lineage_source_model": lineage,
                "prototype_feature_mode": feature_mode,
                "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                "prototype_distance_metric": defaults["prototype_distance_metric"],
                "include_l2_distance": defaults["include_l2_distance"],
                "include_same_cancer_gap": defaults["include_same_cancer_gap"],
                "include_initialized_flag": defaults["include_initialized_flag"],
                "proto_feature_scaler": defaults["proto_feature_scaler"],
                "combined_latent_dir": feature_dir,
                "random_seed": random_seed,
                "status": "pending",
            }
        )

        model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
        ms_row = {
            "ID": job_key,
            "model_type": "VAE",
            "result_folder": feature_dir if feature_mode != "none" else checkpoint_dir,
            "selection_rank": int(model.get("selection_rank", 1)),
            "prototype_feature_mode": feature_mode,
            "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
            "source_model_id": pool_id,
            "source_round": source_round,
            "round15_branch": branch,
            "route_id": route_id,
            "lineage_source_model": lineage,
            "random_seed": random_seed,
        }
        pd.DataFrame([ms_row]).to_csv(model_select_path, index=False)

        for combo in combos:
            ft_job_id = f"ft_{job_key}_c{combo['combo_id']:02d}"
            finetune_rows.append(
                {
                    "job_id": ft_job_id,
                    "model_id": job_key,
                    "source_model_id": pool_id,
                    "source_round": source_round,
                    "round15_branch": branch,
                    "route_id": route_id,
                    "lineage_source_model": lineage,
                    "feature_mode": feature_mode,
                    "prototype_feature_mode": feature_mode,
                    "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                    "combined_latent_dir": feature_dir if feature_mode != "none" else checkpoint_dir,
                    "pretrain_result_dir": checkpoint_dir,
                    "model_select_path": model_select_path,
                    "finetune_config_path": finetune_config,
                    "combo_id": combo["combo_id"],
                    "result_dir": os.path.join(outdir, "finetune", job_key, f"combo_{combo['combo_id']:02d}"),
                    "random_seed": random_seed,
                    "status": "pending",
                    "start_time": "",
                    "end_time": "",
                    "error_message": "",
                }
            )


def build_round15_pretrain_configs(settings_path: str, outdir: str, force: bool = False) -> Optional[str]:
    settings = load_json(resolve_path(settings_path))
    cfg15c = settings.get("round15c_ultra_low_late_vicreg", {})
    if not cfg15c.get("enabled", True):
        return None

    outdir = resolve_path(outdir)
    config_dir = os.path.join(outdir, "configs")
    manifest_dir = os.path.join(outdir, "manifests")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    manifest_path = os.path.join(manifest_dir, "pretrain_sweep_manifest.csv")
    if os.path.exists(manifest_path) and not force:
        return manifest_path

    round12_root = resolve_path(settings["round12_root"])
    source_model = str(cfg15c.get("source_model", cfg15c.get("source_route", "exp_008")))
    if not source_model.startswith("exp_"):
        source_model = f"exp_{source_model}"
    route = {
        "route_id": cfg15c.get("route_id", "exp008_proto_response_route"),
        "source_model": source_model,
        "source_round": cfg15c.get("source_round", "round12"),
        "round14_branch": "15C",
        "description": "Round 15 ultra-low / late VICReg rescue on exp_008 route",
    }
    branch = str(cfg15c.get("round15_branch", "15C"))
    vicreg_settings = cfg15c.get("vicreg_settings", [])
    schedules = cfg15c.get("schedules", [{"start": 60, "full": 120}, {"start": 90, "full": 150}])
    seeds = [int(s) for s in cfg15c.get("seeds", [101, 202, 303])]

    baseline_params, checkpoint_dir = _resolve_pretrain_params(round12_root, source_model)
    base = _base_round14_params(baseline_params, route, settings)
    base["round"] = "round15"
    base["round15_branch"] = branch

    generated_at = datetime.now(timezone.utc).isoformat()
    metadata_base = {
        "round": "round15",
        "generated_at": generated_at,
        "route_id": route["route_id"],
        "source_model": source_model,
        "source_checkpoint_dir": checkpoint_dir,
        "settings_path": os.path.relpath(settings_path, PROJECT_ROOT),
        "round15_branch": branch,
    }

    rows: List[dict] = []
    job_idx = 0

    def next_exp_id() -> str:
        nonlocal job_idx
        job_idx += 1
        return f"exp_{job_idx:03d}"

    for setting in vicreg_settings:
        lam_var = float(setting.get("lambda_tumor_var", 0.0))
        lam_cov = float(setting.get("lambda_tumor_cov", 0.0))
        for sched in schedules:
            start = int(sched["start"])
            full = int(sched["full"])
            for seed in seeds:
                params = _apply_vicreg(base, branch, lam_var, lam_cov, start, full, seed)
                params["round"] = "round15"
                params["round15_branch"] = branch
                exp_id = next_exp_id()
                job_id = (
                    f"r15{branch}_{source_model}_vv{_lam_tag(lam_var)}_{_lam_tag(lam_cov)}"
                    f"_s{start}_f{full}_seed{seed}"
                )
                config_path = os.path.join(config_dir, f"{job_id}.json")
                rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                result_dir = os.path.join(outdir, "pretrain", exp_id)
                payload = {
                    "_metadata": {**metadata_base, "branch": branch, "job_id": job_id},
                    "round15_repro_rescue": True,
                    "pretrain_param_combinations": [params],
                }
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                row = _manifest_row(
                    job_id,
                    exp_id,
                    rel_config,
                    os.path.relpath(result_dir, PROJECT_ROOT),
                    params,
                    route["route_id"],
                    source_model,
                    branch,
                )
                row["round"] = "round15"
                row["round15_branch"] = branch
                rows.append(row)

    manifest_df = pd.DataFrame(rows)
    for col in PRETRAIN_MANIFEST_COLUMNS_ROUND15:
        if col not in manifest_df.columns:
            manifest_df[col] = ""
    manifest_df = manifest_df[PRETRAIN_MANIFEST_COLUMNS_ROUND15]
    manifest_df.to_csv(manifest_path, index=False)

    meta = {
        "round": "round15",
        "generated_at": generated_at,
        "n_pretrain_jobs": len(rows),
        "branch": branch,
        "references": settings.get("references", {}),
    }
    with open(os.path.join(outdir, "round15_build_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(rows)} Round 15C pretrain jobs -> {manifest_path}")
    return manifest_path


def build_round15_finetune_manifest(
    settings_path: str,
    outdir: str,
    selection_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    settings = load_json(resolve_path(settings_path))
    outdir = resolve_path(outdir)
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

    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round15_compact_features.json")
    combos = _expand_finetune_combinations(finetune_config)
    proto_rows: List[dict] = []
    finetune_rows: List[dict] = []
    pool_rows: List[Dict[str, Any]] = []

    cfg15a = settings.get("round15a_repro", {})
    if cfg15a.get("enabled", True):
        round12_root = resolve_path(settings["round12_root"])
        source_model = str(cfg15a.get("source_model", "exp_008"))
        feature_modes = list(cfg15a.get("feature_modes", list(COMPACT_FEATURE_MODES)))
        seeds = [int(s) for s in cfg15a.get("seeds", [101, 202, 303, 404, 505])]
        try:
            checkpoint = _resolve_checkpoint(round12_root, source_model)
            for seed in seeds:
                model = {
                    "pool_model_id": source_model,
                    "role": "round13_best_5seed_repro",
                    "source_round": str(cfg15a.get("source_round", "round12")),
                    "checkpoint_dir": checkpoint,
                    "route_id": "exp008_proto_response_route",
                    "lineage_source_model": source_model,
                    "round15_branch": "15A",
                }
                _append_finetune_block(
                    finetune_rows,
                    proto_rows,
                    model_select_dir,
                    outdir,
                    model,
                    feature_modes,
                    finetune_config,
                    combos,
                    branch="A",
                    random_seed=seed,
                    seed_suffix=f"s{seed}",
                )
            pool_rows.append({**model, "seeds": seeds})
        except FileNotFoundError as exc:
            warnings.warn(str(exc))

    cfg15b = settings.get("round15b_forced_exp008_route", {})
    if cfg15b.get("enabled", True):
        b_pool, pool_warnings = resolve_round15b_model_pool(settings)
        for msg in pool_warnings:
            warnings.warn(msg)
        feature_modes = list(cfg15b.get("feature_modes", list(COMPACT_FEATURE_MODES)))
        for idx, model in enumerate(b_pool, start=1):
            model = dict(model)
            model["selection_rank"] = idx
            _append_finetune_block(
                finetune_rows,
                proto_rows,
                model_select_dir,
                outdir,
                model,
                feature_modes,
                finetune_config,
                combos,
                branch="B",
                random_seed=42,
            )
            pool_rows.append(model)

    if selection_path:
        selection_path = resolve_path(selection_path)
        if os.path.isfile(selection_path):
            selection = pd.read_csv(selection_path)
            id_col = "ID" if "ID" in selection.columns else "model_id"
            folder_col = "result_folder" if "result_folder" in selection.columns else "pretrain_result_dir"
            feature_modes = list(settings.get("response_feature_modes", list(COMPACT_FEATURE_MODES)))
            for rank, (_, sel_row) in enumerate(selection.iterrows(), start=1):
                exp_id = str(sel_row[id_col])
                checkpoint_dir = resolve_path(str(sel_row[folder_col]))
                model = {
                    "pool_model_id": exp_id,
                    "role": str(sel_row.get("round15_selection_group", "round15c_rescue")),
                    "source_round": "round15",
                    "checkpoint_dir": checkpoint_dir,
                    "route_id": str(sel_row.get("route_id", sel_row.get("round15_route_id", "exp008_proto_response_route"))),
                    "lineage_source_model": str(sel_row.get("source_model", sel_row.get("lineage_source_model", "exp_008"))),
                    "selection_rank": rank,
                    "round15_branch": "15C",
                }
                _append_finetune_block(
                    finetune_rows,
                    proto_rows,
                    model_select_dir,
                    outdir,
                    model,
                    feature_modes,
                    finetune_config,
                    combos,
                    branch="C",
                    random_seed=int(sel_row.get("random_seed", 42)),
                )
                pool_rows.append(model)

    pool_path = os.path.join(manifests_dir, "model_pool.csv")
    pd.DataFrame(pool_rows).to_csv(pool_path, index=False)
    pd.DataFrame(proto_rows).to_csv(proto_manifest_path, index=False)
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest_path, index=False)

    meta = {
        "round": "round15",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_proto_jobs": len(proto_rows),
        "n_finetune_jobs": len(finetune_rows),
        "n_pool_models": len(pool_rows),
        "has_15c_selection": bool(selection_path and os.path.isfile(resolve_path(selection_path))),
        "feature_modes": list(settings.get("response_feature_modes", list(COMPACT_FEATURE_MODES))),
    }
    with open(os.path.join(outdir, "round15_finetune_build_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote model pool ({len(pool_rows)} entries) -> {pool_path}")
    print(f"Wrote proto feature manifest ({len(proto_rows)} rows) -> {proto_manifest_path}")
    print(f"Wrote finetune manifest ({len(finetune_rows)} jobs) -> {finetune_manifest_path}")
    return {
        "model_pool": pool_path,
        "proto_feature_manifest": proto_manifest_path,
        "finetune_dispatch_manifest": finetune_manifest_path,
    }


def build_round15_all(settings_path: str, outdir: str, force: bool = False) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    pretrain_path = build_round15_pretrain_configs(settings_path, outdir, force=force)
    if pretrain_path:
        outputs["pretrain_sweep_manifest"] = pretrain_path
    outputs.update(build_round15_finetune_manifest(settings_path, outdir, selection_path=None, force=force))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 15 repro + exp_008 rescue configs")
    parser.add_argument("--settings", default="config/round15_repro_rescue_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round15_repro_rescue")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build-pretrain", action="store_true")
    parser.add_argument("--build-finetune-manifest", action="store_true")
    parser.add_argument(
        "--selection",
        default=None,
        help="Optional pretrain_top10.csv for Round 15C downstream rebuild",
    )
    args = parser.parse_args()

    if args.build_pretrain:
        build_round15_pretrain_configs(args.settings, args.outdir, force=args.force)
    elif args.build_finetune_manifest:
        build_round15_finetune_manifest(
            args.settings,
            args.outdir,
            selection_path=args.selection,
            force=args.force,
        )
    else:
        build_round15_all(args.settings, args.outdir, force=args.force)


if __name__ == "__main__":
    main()
