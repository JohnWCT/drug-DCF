#!/usr/bin/env python3
"""Build Round 10 Conditional ADV configs and pretrain manifest."""

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

from tools.round9_diagnostics_common import load_json, resolve_path

ROUND10_DISABLED_LOSSES = {
    "lambda_proto": 0,
    "lambda_class_gap": 0,
    "lambda_tumor_topology": 0,
    "lambda_tumor_supcon": 0,
    "lambda_subspace_ortho": 0,
    "use_tumor_subspace": False,
}

ROUND10B_VICREG_ZERO = {
    "lambda_tumor_var": 0,
    "lambda_tumor_cov": 0,
}

MANIFEST_COLUMNS = [
    "job_id",
    "exp_id",
    "config_path",
    "result_dir",
    "round",
    "round10_branch",
    "source_baseline_exp_id",
    "conditional_adv_enabled",
    "conditional_adv_mode",
    "cancer_condition_dim",
    "lambda_cond_adv",
    "cond_adv_start_epoch",
    "cond_adv_full_epoch",
    "global_adv_mode",
    "lambda_global_adv_multiplier",
    "random_seed",
    "status",
]


def _read_baseline_params(checkpoint_dir: str) -> dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    params_path = os.path.join(checkpoint_dir, "params.json")
    if os.path.exists(params_path):
        payload = load_json(params_path)
        params = copy.deepcopy(payload.get("params", payload))
        if params:
            return params
    summary_path = os.path.join(checkpoint_dir, "run_summary.json")
    if os.path.exists(summary_path):
        payload = load_json(summary_path)
        params = copy.deepcopy(payload.get("params", {}))
        if params:
            return params
    raise FileNotFoundError(
        f"Missing params.json and run_summary.params for baseline: {checkpoint_dir}"
    )


def _lam_tag(value: float) -> str:
    return f"{value:.4f}".replace(".", "").lstrip("0") or "0"


def _write_config(path: str, params: dict, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_metadata": metadata,
        "round10_cond_adv": True,
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
        "round": params.get("round", "round10"),
        "round10_branch": params.get("round10_branch", ""),
        "source_baseline_exp_id": params.get("source_baseline_exp_id", ""),
        "conditional_adv_enabled": params.get("conditional_adv_enabled", False),
        "conditional_adv_mode": params.get("conditional_adv_mode", "none"),
        "cancer_condition_dim": params.get("cancer_condition_dim", 0),
        "lambda_cond_adv": params.get("lambda_cond_adv", 0.0),
        "cond_adv_start_epoch": params.get("cond_adv_start_epoch", 0),
        "cond_adv_full_epoch": params.get("cond_adv_full_epoch", 0),
        "global_adv_mode": params.get("global_adv_mode", "baseline_global_only"),
        "lambda_global_adv_multiplier": params.get("lambda_global_adv_multiplier", 1.0),
        "random_seed": params.get("random_seed", 0),
        "status": "pending",
    }


def _resolve_baseline_row(
    resolved_baselines_path: str,
    primary_baseline_exp_id: str,
    baseline_config: str | None,
) -> pd.Series:
    if baseline_config:
        checkpoint_dir = resolve_path(baseline_config)
        if os.path.isdir(checkpoint_dir):
            return pd.Series(
                {
                    "exp_id": primary_baseline_exp_id,
                    "checkpoint_dir": checkpoint_dir,
                    "resolved": True,
                }
            )
        params = load_json(baseline_config)
        ckpt = params.get("checkpoint_dir") or params.get("params", {}).get("checkpoint_dir")
        if ckpt:
            return pd.Series(
                {
                    "exp_id": primary_baseline_exp_id,
                    "checkpoint_dir": resolve_path(ckpt),
                    "resolved": True,
                }
            )
        raise FileNotFoundError(f"baseline-config has no checkpoint_dir: {baseline_config}")

    resolved_path = resolve_path(resolved_baselines_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Round 9 resolved baselines missing: {resolved_path}")
    resolved = pd.read_csv(resolved_path)
    row = resolved[
        (resolved["exp_id"].astype(str) == primary_baseline_exp_id)
        & (resolved["resolved"] == True)  # noqa: E712
    ]
    if row.empty:
        raise FileNotFoundError(
            f"Primary baseline {primary_baseline_exp_id} not resolved in {resolved_path}"
        )
    return row.iloc[0]


def build_round10_configs(
    settings_path: str,
    outdir: str,
    force: bool = False,
    round9_summary: str | None = None,
    round9_model_summary: str | None = None,
    baseline_config: str | None = None,
    primary_baseline_exp_id: str | None = None,
) -> str:
    settings = load_json(resolve_path(settings_path))
    outdir = resolve_path(outdir)
    config_dir = os.path.join(outdir, "configs")
    manifest_dir = os.path.join(outdir, "manifests")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    r9_summary_path = resolve_path(round9_summary or settings["round9_summary"])
    if not os.path.exists(r9_summary_path):
        raise FileNotFoundError(f"Round 9 summary missing (fail fast): {r9_summary_path}")
    r9_model_path = resolve_path(round9_model_summary or settings["round9_model_summary"])
    if not os.path.exists(r9_model_path):
        raise FileNotFoundError(f"Round 9 model summary missing (fail fast): {r9_model_path}")

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
    branches = settings.get("branch_design", {})
    cond_cfg = settings.get("conditional_adv", {})
    weak_cfg = settings.get("weak_global_guard", {})
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
        "round": "round10",
        "generated_at": generated_at,
        "source_baseline_exp_id": baseline_id,
        "source_checkpoint_dir": str(baseline_row["checkpoint_dir"]),
    }

    if branches.get("include_10A_global_repro", True):
        for seed in seeds:
            exp_id = next_exp_id()
            params = copy.deepcopy(baseline_params)
            params.update(
                {
                    "round": "round10",
                    "round10_branch": "10A_global_adv_repro",
                    "source_baseline_exp_id": baseline_id,
                    "conditional_adv_enabled": False,
                    "conditional_adv_mode": "none",
                    "lambda_cond_adv": 0.0,
                    "global_adv_mode": "baseline_global_only",
                    "lambda_global_adv_multiplier": 1.0,
                    "random_seed": int(seed),
                }
            )
            config_name = f"round10A_{baseline_id}_global_seed{seed}.json"
            config_path = os.path.join(config_dir, config_name)
            rel_config = os.path.relpath(config_path, PROJECT_ROOT)
            result_dir = os.path.join(outdir, "pretrain", exp_id)
            _write_config(config_path, params, {**metadata_base, "branch": "10A"})
            rows.append(
                _manifest_row(
                    f"r10A_{baseline_id}_seed{seed}",
                    exp_id,
                    rel_config,
                    os.path.relpath(result_dir, PROJECT_ROOT),
                    params,
                )
            )

    if branches.get("include_10B_conditional_replacement", True):
        for lam in cond_cfg.get("lambdas", []):
            for dim in cond_cfg.get("condition_dims", []):
                for sched in cond_cfg.get("schedules", []):
                    for seed in seeds:
                        exp_id = next_exp_id()
                        params = copy.deepcopy(baseline_params)
                        params.update(ROUND10_DISABLED_LOSSES)
                        params.update(ROUND10B_VICREG_ZERO)
                        start = int(sched["start"])
                        full = int(sched["full"])
                        params.update(
                            {
                                "round": "round10",
                                "round10_branch": "10B_conditional_replacement",
                                "source_baseline_exp_id": baseline_id,
                                "conditional_adv_enabled": True,
                                "conditional_adv_mode": cond_cfg.get("mode", "cancer_embedding"),
                                "cancer_condition_dim": int(dim),
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
                            f"round10B_{baseline_id}_cond_replace_lam{_lam_tag(lam)}"
                            f"_dim{dim}_s{start}_f{full}_seed{seed}.json"
                        )
                        config_path = os.path.join(config_dir, config_name)
                        rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                        result_dir = os.path.join(outdir, "pretrain", exp_id)
                        _write_config(config_path, params, {**metadata_base, "branch": "10B"})
                        rows.append(
                            _manifest_row(
                                f"r10B_lam{_lam_tag(lam)}_d{dim}_s{start}_f{full}_seed{seed}",
                                exp_id,
                                rel_config,
                                os.path.relpath(result_dir, PROJECT_ROOT),
                                params,
                            )
                        )

    if branches.get("include_10C_conditional_plus_weak_global", True) and weak_cfg.get(
        "enabled", True
    ):
        mult = float(weak_cfg.get("lambda_global_adv_multiplier", 0.25))
        for lam in weak_cfg.get("lambdas", []):
            for dim in weak_cfg.get("condition_dims", [16]):
                for sched in weak_cfg.get("schedules", []):
                    for seed in seeds:
                        exp_id = next_exp_id()
                        params = copy.deepcopy(baseline_params)
                        params.update(ROUND10_DISABLED_LOSSES)
                        params.update(ROUND10B_VICREG_ZERO)
                        start = int(sched["start"])
                        full = int(sched["full"])
                        params.update(
                            {
                                "round": "round10",
                                "round10_branch": "10C_conditional_plus_weak_global",
                                "source_baseline_exp_id": baseline_id,
                                "conditional_adv_enabled": True,
                                "conditional_adv_mode": cond_cfg.get("mode", "cancer_embedding"),
                                "cancer_condition_dim": int(dim),
                                "lambda_cond_adv": float(lam),
                                "cond_adv_start_epoch": start,
                                "cond_adv_full_epoch": full,
                                "global_adv_mode": "conditional_plus_weak_global",
                                "lambda_global_adv_multiplier": mult,
                                "cond_critic_hidden_dims": hidden_dims,
                                "cond_critic_dropout": dropout,
                                "random_seed": int(seed),
                            }
                        )
                        config_name = (
                            f"round10C_{baseline_id}_cond_weakglobal_lam{_lam_tag(lam)}"
                            f"_dim{dim}_s{start}_f{full}_seed{seed}.json"
                        )
                        config_path = os.path.join(config_dir, config_name)
                        rel_config = os.path.relpath(config_path, PROJECT_ROOT)
                        result_dir = os.path.join(outdir, "pretrain", exp_id)
                        _write_config(config_path, params, {**metadata_base, "branch": "10C"})
                        rows.append(
                            _manifest_row(
                                f"r10C_lam{_lam_tag(lam)}_d{dim}_s{start}_f{full}_seed{seed}",
                                exp_id,
                                rel_config,
                                os.path.relpath(result_dir, PROJECT_ROOT),
                                params,
                            )
                        )

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(manifest_path, index=False)

    sweep_json_path = os.path.join(PROJECT_ROOT, "config/pretrain_sweeps/vaewc_round10_cond_adv.json")
    os.makedirs(os.path.dirname(sweep_json_path), exist_ok=True)
    sweep_meta: Dict[str, Any] = {
        "round": "round10",
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
    parser = argparse.ArgumentParser(description="Build Round 10 Conditional ADV configs")
    parser.add_argument("--settings", default="config/round10_cond_adv_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round10_cond_adv")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--round9-summary", default=None)
    parser.add_argument("--round9-model-summary", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--primary-baseline-exp-id", default=None)
    args = parser.parse_args()

    manifest = build_round10_configs(
        settings_path=args.settings,
        outdir=args.outdir,
        force=args.force,
        round9_summary=args.round9_summary,
        round9_model_summary=args.round9_model_summary,
        baseline_config=args.baseline_config,
        primary_baseline_exp_id=args.primary_baseline_exp_id,
    )
    df = pd.read_csv(manifest)
    print(f"Wrote {len(df)} jobs to {manifest}")
    print(df["round10_branch"].value_counts().to_string())


if __name__ == "__main__":
    main()
