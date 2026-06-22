#!/usr/bin/env python3
"""Build Round 11 configs: 10C stabilization + SmoothL1 reconstruction ablation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.reconstruction_losses import reconstruction_loss_defaults
from tools.round10_config_builder import (
    ROUND10B_VICREG_ZERO,
    ROUND10_DISABLED_LOSSES,
    _lam_tag,
    _read_baseline_params,
    _resolve_baseline_row,
)
from tools.round9_diagnostics_common import load_json, resolve_path

MANIFEST_COLUMNS = [
    "job_id",
    "exp_id",
    "config_path",
    "result_dir",
    "round",
    "round11_branch",
    "source_baseline_exp_id",
    "conditional_adv_enabled",
    "conditional_adv_mode",
    "cancer_condition_dim",
    "lambda_cond_adv",
    "cond_adv_start_epoch",
    "cond_adv_full_epoch",
    "global_adv_mode",
    "lambda_global_adv_multiplier",
    "reconstruction_loss_type",
    "smooth_l1_beta",
    "reconstruction_loss_scale",
    "random_seed",
    "status",
]


def _write_config(path: str, params: dict, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_metadata": metadata,
        "round11_stability_recon": True,
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
        "round": params.get("round", "round11"),
        "round11_branch": params.get("round11_branch", ""),
        "source_baseline_exp_id": params.get("source_baseline_exp_id", ""),
        "conditional_adv_enabled": params.get("conditional_adv_enabled", False),
        "conditional_adv_mode": params.get("conditional_adv_mode", "none"),
        "cancer_condition_dim": params.get("cancer_condition_dim", 0),
        "lambda_cond_adv": params.get("lambda_cond_adv", 0.0),
        "cond_adv_start_epoch": params.get("cond_adv_start_epoch", 0),
        "cond_adv_full_epoch": params.get("cond_adv_full_epoch", 0),
        "global_adv_mode": params.get("global_adv_mode", "baseline_global_only"),
        "lambda_global_adv_multiplier": params.get("lambda_global_adv_multiplier", 1.0),
        "reconstruction_loss_type": params.get("reconstruction_loss_type", "mse"),
        "smooth_l1_beta": params.get("smooth_l1_beta", 1.0),
        "reconstruction_loss_scale": params.get("reconstruction_loss_scale", 1.0),
        "random_seed": params.get("random_seed", 0),
        "status": "pending",
    }


def _base_round11_params(baseline_params: dict, baseline_id: str) -> dict:
    params = copy.deepcopy(baseline_params)
    params.update(ROUND10_DISABLED_LOSSES)
    params.update(ROUND10B_VICREG_ZERO)
    params.update(reconstruction_loss_defaults())
    params.update(
        {
            "round": "round11",
            "source_baseline_exp_id": baseline_id,
        }
    )
    return params


def build_round11_configs(
    settings_path: str,
    outdir: str,
    force: bool = False,
    baseline_config: str | None = None,
    primary_baseline_exp_id: str | None = None,
) -> str:
    settings = load_json(resolve_path(settings_path))
    outdir = resolve_path(outdir)
    config_dir = os.path.join(outdir, "configs")
    manifest_dir = os.path.join(outdir, "manifests")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    baseline_id = primary_baseline_exp_id or settings["primary_baseline_exp_id"]
    baseline_row = _resolve_baseline_row(
        settings.get("resolved_baselines", ""),
        baseline_id,
        baseline_config,
    )
    baseline_params = _read_baseline_params(str(baseline_row["checkpoint_dir"]))

    manifest_path = os.path.join(manifest_dir, "pretrain_sweep_manifest.csv")
    if os.path.exists(manifest_path) and not force:
        return manifest_path

    seeds: List[int] = [int(s) for s in settings.get("seeds", [101, 202, 303])]
    cond_cfg = settings.get("conditional_adv", {})
    hidden_dims = cond_cfg.get("hidden_dims", [128, 64])
    dropout = float(cond_cfg.get("dropout", 0.1))

    rows: List[dict] = []
    job_idx = 0
    generated_at = datetime.now(timezone.utc).isoformat()

    def next_exp_id() -> str:
        nonlocal job_idx
        job_idx += 1
        return f"exp_{job_idx:03d}"

    metadata_base = {
        "round": "round11",
        "generated_at": generated_at,
        "source_baseline_exp_id": baseline_id,
        "source_checkpoint_dir": str(baseline_row["checkpoint_dir"]),
    }

    b_cfg = settings.get("round11b_condadv_stabilization", {})
    if b_cfg.get("enabled", True):
        for lam in b_cfg.get("lambda_cond_adv", []):
            for sched in b_cfg.get("schedules", []):
                for mult in b_cfg.get("lambda_global_adv_multiplier", [0.25]):
                    for seed in seeds:
                        for dim in b_cfg.get("condition_dims", [16]):
                            exp_id = next_exp_id()
                            params = _base_round11_params(baseline_params, baseline_id)
                            start = int(sched["start"])
                            full = int(sched["full"])
                            params.update(
                                {
                                    "round11_branch": "11B_10C_stabilization",
                                    "conditional_adv_enabled": True,
                                    "conditional_adv_mode": cond_cfg.get("mode", "cancer_embedding"),
                                    "cancer_condition_dim": int(dim),
                                    "lambda_cond_adv": float(lam),
                                    "cond_adv_start_epoch": start,
                                    "cond_adv_full_epoch": full,
                                    "global_adv_mode": b_cfg.get(
                                        "global_adv_mode", "conditional_plus_weak_global"
                                    ),
                                    "lambda_global_adv_multiplier": float(mult),
                                    "cond_critic_hidden_dims": hidden_dims,
                                    "cond_critic_dropout": dropout,
                                    "random_seed": int(seed),
                                }
                            )
                            config_name = (
                                f"round11B_10C_lam{_lam_tag(lam)}_m{str(mult).replace('.', '')}"
                                f"_d{dim}_s{start}_f{full}_seed{seed}.json"
                            )
                            config_path = os.path.join(config_dir, config_name)
                            rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                            result_dir = os.path.join(outdir, "pretrain", exp_id)
                            _write_config(config_path, params, {**metadata_base, "branch": "11B"})
                            rows.append(
                                _manifest_row(
                                    f"r11B_10C_lam{_lam_tag(lam)}_m{str(mult).replace('.', '')}"
                                    f"_d{dim}_s{start}_f{full}_seed{seed}",
                                    exp_id,
                                    rel_config,
                                    os.path.relpath(result_dir, PROJECT_ROOT),
                                    params,
                                )
                            )

        if b_cfg.get("include_small_10b_control", True):
            control_lams = [0.0001, 0.0003]
            control_scheds = [{"start": 20, "full": 60}, {"start": 20, "full": 90}]
            for lam in control_lams:
                for sched in control_scheds:
                    for seed in seeds:
                        exp_id = next_exp_id()
                        params = _base_round11_params(baseline_params, baseline_id)
                        start = int(sched["start"])
                        full = int(sched["full"])
                        params.update(
                            {
                                "round11_branch": "11B_10B_control",
                                "conditional_adv_enabled": True,
                                "conditional_adv_mode": cond_cfg.get("mode", "cancer_embedding"),
                                "cancer_condition_dim": 16,
                                "lambda_cond_adv": float(lam),
                                "cond_adv_start_epoch": start,
                                "cond_adv_full_epoch": full,
                                "global_adv_mode": "conditional_replacement",
                                "lambda_global_adv_multiplier": 0.0,
                                "cond_critic_hidden_dims": hidden_dims,
                                "cond_critic_dropout": dropout,
                                "random_seed": int(seed),
                            }
                        )
                        config_name = (
                            f"round11B_10B_ctrl_lam{_lam_tag(lam)}_s{start}_f{full}_seed{seed}.json"
                        )
                        config_path = os.path.join(config_dir, config_name)
                        rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                        result_dir = os.path.join(outdir, "pretrain", exp_id)
                        _write_config(config_path, params, {**metadata_base, "branch": "11B_control"})
                        rows.append(
                            _manifest_row(
                                f"r11B_10B_lam{_lam_tag(lam)}_s{start}_f{full}_seed{seed}",
                                exp_id,
                                rel_config,
                                os.path.relpath(result_dir, PROJECT_ROOT),
                                params,
                            )
                        )

    c_cfg = settings.get("round11c_reconstruction_ablation", {})
    if c_cfg.get("enabled", True):
        recon_types = c_cfg.get("reconstruction_loss_types", ["mse", "smooth_l1"])
        betas = c_cfg.get("smooth_l1_beta", [0.5, 1.0])
        hybrid_alpha = float(c_cfg.get("hybrid_reconstruction_alpha", [0.5])[0])

        for loss_type in recon_types:
            beta_list = [1.0] if loss_type == "mse" else betas
            for beta in beta_list:
                for seed in seeds:
                    exp_id = next_exp_id()
                    params = _base_round11_params(baseline_params, baseline_id)
                    params.update(
                        {
                            "round11_branch": "11C_global_recon_control",
                            "conditional_adv_enabled": False,
                            "conditional_adv_mode": "none",
                            "lambda_cond_adv": 0.0,
                            "global_adv_mode": "baseline_global_only",
                            "lambda_global_adv_multiplier": 1.0,
                            "reconstruction_loss_type": loss_type,
                            "smooth_l1_beta": float(beta),
                            "hybrid_reconstruction_alpha": hybrid_alpha,
                            "random_seed": int(seed),
                        }
                    )
                    beta_tag = "" if loss_type == "mse" else f"_b{str(beta).replace('.', '')}"
                    config_name = f"round11C_global_{loss_type}{beta_tag}_seed{seed}.json"
                    config_path = os.path.join(config_dir, config_name)
                    rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                    result_dir = os.path.join(outdir, "pretrain", exp_id)
                    _write_config(config_path, params, {**metadata_base, "branch": "11C_global"})
                    rows.append(
                        _manifest_row(
                            f"r11C_global_{loss_type}{beta_tag}_seed{seed}",
                            exp_id,
                            rel_config,
                            os.path.relpath(result_dir, PROJECT_ROOT),
                            params,
                        )
                    )

        c10_lams = [0.0003, 0.001]
        c10_scheds = [{"start": 20, "full": 60}, {"start": 20, "full": 90}]
        for lam in c10_lams:
            for sched in c10_scheds:
                for loss_type in ["mse", "smooth_l1"]:
                    beta_list = [1.0] if loss_type == "mse" else [0.5, 1.0]
                    for beta in beta_list:
                        for seed in seeds:
                            exp_id = next_exp_id()
                            params = _base_round11_params(baseline_params, baseline_id)
                            start = int(sched["start"])
                            full = int(sched["full"])
                            params.update(
                                {
                                    "round11_branch": "11C_10C_recon_ablation",
                                    "conditional_adv_enabled": True,
                                    "conditional_adv_mode": cond_cfg.get("mode", "cancer_embedding"),
                                    "cancer_condition_dim": 16,
                                    "lambda_cond_adv": float(lam),
                                    "cond_adv_start_epoch": start,
                                    "cond_adv_full_epoch": full,
                                    "global_adv_mode": "conditional_plus_weak_global",
                                    "lambda_global_adv_multiplier": 0.25,
                                    "cond_critic_hidden_dims": hidden_dims,
                                    "cond_critic_dropout": dropout,
                                    "reconstruction_loss_type": loss_type,
                                    "smooth_l1_beta": float(beta),
                                    "hybrid_reconstruction_alpha": hybrid_alpha,
                                    "random_seed": int(seed),
                                }
                            )
                            beta_tag = "" if loss_type == "mse" else f"_b{str(beta).replace('.', '')}"
                            config_name = (
                                f"round11C_10C_lam{_lam_tag(lam)}_{loss_type}{beta_tag}"
                                f"_s{start}_f{full}_seed{seed}.json"
                            )
                            config_path = os.path.join(config_dir, config_name)
                            rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                            result_dir = os.path.join(outdir, "pretrain", exp_id)
                            _write_config(config_path, params, {**metadata_base, "branch": "11C_10C"})
                            rows.append(
                                _manifest_row(
                                    f"r11C_10C_lam{_lam_tag(lam)}_{loss_type}{beta_tag}"
                                    f"_s{start}_f{full}_seed{seed}",
                                    exp_id,
                                    rel_config,
                                    os.path.relpath(result_dir, PROJECT_ROOT),
                                    params,
                                )
                            )

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(manifest_path, index=False)

    sweep_json_path = os.path.join(PROJECT_ROOT, "config/pretrain_sweeps/vaewc_round11_smoothl1_condadv.json")
    os.makedirs(os.path.dirname(sweep_json_path), exist_ok=True)
    sweep_meta: Dict[str, Any] = {
        "round": "round11",
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
    parser = argparse.ArgumentParser(description="Build Round 11 configs")
    parser.add_argument("--settings", default="config/round11_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round11_stability_recon")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--primary-baseline-exp-id", default=None)
    args = parser.parse_args()

    manifest = build_round11_configs(
        settings_path=args.settings,
        outdir=args.outdir,
        force=args.force,
        baseline_config=args.baseline_config,
        primary_baseline_exp_id=args.primary_baseline_exp_id,
    )
    df = pd.read_csv(manifest)
    print(f"Wrote {len(df)} jobs to {manifest}")
    if "round11_branch" in df.columns:
        print(df["round11_branch"].value_counts().to_string())


if __name__ == "__main__":
    main()
