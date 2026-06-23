#!/usr/bin/env python3
"""Build Round 12 configs: source-anchor EMA prototype alignment on Round 11 exp_035."""

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

from tools.reconstruction_losses import reconstruction_loss_defaults
from tools.round10_config_builder import ROUND10B_VICREG_ZERO, ROUND10_DISABLED_LOSSES, _lam_tag
from tools.round9_diagnostics_common import load_json, resolve_path

MANIFEST_COLUMNS = [
    "job_id",
    "exp_id",
    "config_path",
    "result_dir",
    "round",
    "round12_branch",
    "source_baseline_exp_id",
    "conditional_adv_enabled",
    "global_adv_mode",
    "lambda_cond_adv",
    "lambda_global_adv_multiplier",
    "source_anchor_proto_enabled",
    "lambda_proto_align",
    "proto_align_metric",
    "proto_align_start_epoch",
    "proto_align_full_epoch",
    "proto_ema_momentum",
    "reconstruction_loss_type",
    "smooth_l1_beta",
    "random_seed",
    "status",
]


def _write_config(path: str, params: dict, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_metadata": metadata,
        "round12_proto_alignment": True,
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
) -> dict:
    return {
        "job_id": job_id,
        "exp_id": exp_id,
        "config_path": config_path,
        "result_dir": result_dir,
        "round": params.get("round", "round12"),
        "round12_branch": params.get("round12_branch", ""),
        "source_baseline_exp_id": params.get("source_baseline_exp_id", ""),
        "conditional_adv_enabled": params.get("conditional_adv_enabled", False),
        "global_adv_mode": params.get("global_adv_mode", "baseline_global_only"),
        "lambda_cond_adv": params.get("lambda_cond_adv", 0.0),
        "lambda_global_adv_multiplier": params.get("lambda_global_adv_multiplier", 1.0),
        "source_anchor_proto_enabled": params.get("source_anchor_proto_enabled", False),
        "lambda_proto_align": params.get("lambda_proto_align", 0.0),
        "proto_align_metric": params.get("proto_align_metric", "cosine"),
        "proto_align_start_epoch": params.get("proto_align_start_epoch", 0),
        "proto_align_full_epoch": params.get("proto_align_full_epoch", 0),
        "proto_ema_momentum": params.get("proto_ema_momentum", 0.95),
        "reconstruction_loss_type": params.get("reconstruction_loss_type", "mse"),
        "smooth_l1_beta": params.get("smooth_l1_beta", 1.0),
        "random_seed": params.get("random_seed", 0),
        "status": "pending",
    }


def _resolve_round11_baseline(round11_root: str, exp_id: str) -> Tuple[dict, str]:
    round11_root = resolve_path(round11_root)
    pretrain_dir = os.path.join(round11_root, "pretrain", exp_id)
    params_path = os.path.join(pretrain_dir, "params.json")
    if not os.path.isfile(params_path):
        manifest_path = os.path.join(round11_root, "manifests", "pretrain_sweep_manifest.csv")
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
        raise FileNotFoundError(
            f"Round 11 baseline {exp_id} not found under {round11_root} (missing params.json)"
        )
    payload = load_json(params_path)
    params = payload.get("params", payload)
    return copy.deepcopy(params), pretrain_dir


def _base_round12_params(baseline_params: dict, baseline_id: str, settings: dict) -> dict:
    params = copy.deepcopy(baseline_params)
    params.update(ROUND10_DISABLED_LOSSES)
    params.update(ROUND10B_VICREG_ZERO)
    params.update(reconstruction_loss_defaults())
    params.update(settings.get("base_conditional_adv", {}))
    params.update(
        {
            "round": "round12",
            "source_baseline_exp_id": baseline_id,
            "source_anchor_proto_enabled": False,
            "lambda_proto_align": 0.0,
            "proto_align_metric": "cosine",
            "proto_align_start_epoch": 40,
            "proto_align_full_epoch": 90,
            "proto_ema_momentum": 0.95,
            "proto_align_min_count": 2,
            "proto_align_normalize": True,
            "proto_align_update_source_ema": True,
        }
    )
    return params


def _apply_proto_params(params: dict, branch: str, proto_enabled: bool, **overrides) -> dict:
    out = copy.deepcopy(params)
    out["round12_branch"] = branch
    out["source_anchor_proto_enabled"] = bool(proto_enabled)
    out.update(overrides)
    if not proto_enabled:
        out["lambda_proto_align"] = 0.0
    return out


def build_round12_configs(
    settings_path: str,
    outdir: str,
    force: bool = False,
    round11_root: Optional[str] = None,
    primary_baseline_exp_id: Optional[str] = None,
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

    baseline_id = primary_baseline_exp_id or settings["primary_baseline_exp_id"]
    r11_root = resolve_path(round11_root or settings["round11_root"])
    baseline_params, checkpoint_dir = _resolve_round11_baseline(r11_root, baseline_id)

    seeds: List[int] = [int(s) for s in settings.get("seeds", [101, 202, 303])]
    rows: List[dict] = []
    job_idx = 0
    generated_at = datetime.now(timezone.utc).isoformat()

    def next_exp_id() -> str:
        nonlocal job_idx
        job_idx += 1
        return f"exp_{job_idx:03d}"

    metadata_base = {
        "round": "round12",
        "generated_at": generated_at,
        "source_baseline_exp_id": baseline_id,
        "source_checkpoint_dir": checkpoint_dir,
        "round11_root": r11_root,
    }

    base = _base_round12_params(baseline_params, baseline_id, settings)

    def append_job(job_id: str, params: dict, config_name: str, branch_tag: str) -> None:
        exp_id = next_exp_id()
        config_path = os.path.join(config_dir, config_name)
        rel_config = os.path.relpath(config_path, PROJECT_ROOT)
        result_dir = os.path.join(outdir, "pretrain", exp_id)
        _write_config(config_path, params, {**metadata_base, "branch": branch_tag})
        rows.append(
            _manifest_row(
                job_id,
                exp_id,
                rel_config,
                os.path.relpath(result_dir, PROJECT_ROOT),
                params,
            )
        )

    b_cfg = settings.get("round12b_proto_alignment_main", {})
    if b_cfg.get("enabled", True):
        metrics = b_cfg.get("proto_align_metric", ["cosine"])
        lambdas = b_cfg.get("lambda_proto_align", [])
        schedules = b_cfg.get("proto_align_schedules", [])
        momenta = b_cfg.get("proto_ema_momentum", [0.95])
        min_count = int(b_cfg.get("proto_align_min_count", 2))
        recon_types = b_cfg.get("reconstruction_loss_type", ["mse"])

        for lam in lambdas:
            for sched in schedules:
                start = int(sched["start"])
                full = int(sched["full"])
                for momentum in momenta:
                    for metric in metrics:
                        for recon in recon_types:
                            for seed in seeds:
                                params = _apply_proto_params(
                                    base,
                                    "12B_proto_alignment_main",
                                    True,
                                    lambda_proto_align=float(lam),
                                    proto_align_metric=str(metric),
                                    proto_align_start_epoch=start,
                                    proto_align_full_epoch=full,
                                    proto_ema_momentum=float(momentum),
                                    proto_align_min_count=min_count,
                                    reconstruction_loss_type=str(recon),
                                    random_seed=int(seed),
                                )
                                job_id = (
                                    f"r12B_lam{_lam_tag(lam)}_{metric}_s{start}_f{full}"
                                    f"_m{str(momentum).replace('.', '')}_seed{seed}"
                                )
                                config_name = f"{job_id}.json"
                                append_job(job_id, params, config_name, "12B")

        d_cfg = settings.get("round12d_ablation_controls", {})
        if d_cfg.get("include_no_proto_control", True):
            for seed in seeds:
                params = _apply_proto_params(
                    base,
                    "12B_no_proto_control",
                    False,
                    reconstruction_loss_type="mse",
                    random_seed=int(seed),
                )
                job_id = f"r12B_no_proto_seed{seed}"
                append_job(job_id, params, f"{job_id}.json", "12B_control")

    c_cfg = settings.get("round12c_reconstruction_small_branch", {})
    if c_cfg.get("enabled", True):
        recon_types = c_cfg.get("reconstruction_loss_type", [])
        betas = c_cfg.get("smooth_l1_beta", [1.0])
        lambdas = c_cfg.get("lambda_proto_align", [])
        schedules = c_cfg.get("proto_align_schedules", [{"start": 40, "full": 90}])
        momenta = c_cfg.get("proto_ema_momentum", [0.95])
        hybrid_alpha = 0.5

        for loss_type in recon_types:
            beta_list = betas if loss_type != "mse" else [1.0]
            for beta in beta_list:
                for lam in lambdas:
                    for sched in schedules:
                        start = int(sched["start"])
                        full = int(sched["full"])
                        for momentum in momenta:
                            for seed in seeds:
                                params = _apply_proto_params(
                                    base,
                                    "12C_recon_proto",
                                    True,
                                    lambda_proto_align=float(lam),
                                    proto_align_metric="cosine",
                                    proto_align_start_epoch=start,
                                    proto_align_full_epoch=full,
                                    proto_ema_momentum=float(momentum),
                                    reconstruction_loss_type=str(loss_type),
                                    smooth_l1_beta=float(beta),
                                    hybrid_reconstruction_alpha=hybrid_alpha,
                                    random_seed=int(seed),
                                )
                                beta_tag = f"_b{str(beta).replace('.', '')}" if loss_type != "mse" else ""
                                job_id = (
                                    f"r12C_{loss_type}{beta_tag}_lam{_lam_tag(lam)}"
                                    f"_s{start}_f{full}_seed{seed}"
                                )
                                append_job(job_id, params, f"{job_id}.json", "12C")

    d_cfg = settings.get("round12d_ablation_controls", {})
    if d_cfg.get("enabled", True) and d_cfg.get("include_euclidean_metric_small_control", True):
        sched = b_cfg.get("proto_align_schedules", [{"start": 40, "full": 90}])[1] if len(
            b_cfg.get("proto_align_schedules", [])
        ) > 1 else {"start": 40, "full": 90}
        start = int(sched["start"])
        full = int(sched["full"])
        for lam in d_cfg.get("euclidean_lambda_proto_align", [0.0003]):
            for seed in seeds:
                params = _apply_proto_params(
                    base,
                    "12D_euclidean_control",
                    True,
                    lambda_proto_align=float(lam),
                    proto_align_metric="euclidean",
                    proto_align_start_epoch=start,
                    proto_align_full_epoch=full,
                    proto_ema_momentum=0.95,
                    reconstruction_loss_type="mse",
                    random_seed=int(seed),
                )
                job_id = f"r12D_eucl_lam{_lam_tag(lam)}_s{start}_f{full}_seed{seed}"
                append_job(job_id, params, f"{job_id}.json", "12D")

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(manifest_path, index=False)

    sweep_json_path = os.path.join(
        PROJECT_ROOT, "config/pretrain_sweeps/vaewc_round12_proto_alignment.json"
    )
    os.makedirs(os.path.dirname(sweep_json_path), exist_ok=True)
    sweep_meta: Dict[str, Any] = {
        "round": "round12",
        "purpose": settings.get("purpose", ""),
        "generated_at": generated_at,
        "job_count": len(rows),
        "manifest_path": os.path.relpath(manifest_path, PROJECT_ROOT),
        "settings_path": os.path.relpath(settings_path, PROJECT_ROOT),
    }
    with open(sweep_json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_meta, f, indent=2)

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 12 prototype alignment configs")
    parser.add_argument("--settings", default="config/round12_proto_alignment_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round12_proto_alignment")
    parser.add_argument("--round11-root", default=None)
    parser.add_argument("--primary-baseline-exp-id", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_round12_configs(
        settings_path=args.settings,
        outdir=args.outdir,
        force=args.force,
        round11_root=args.round11_root,
        primary_baseline_exp_id=args.primary_baseline_exp_id,
    )
    df = pd.read_csv(manifest)
    print(f"Wrote {len(df)} jobs to {manifest}")
    if "round12_branch" in df.columns:
        print(df["round12_branch"].value_counts().to_string())


if __name__ == "__main__":
    main()
