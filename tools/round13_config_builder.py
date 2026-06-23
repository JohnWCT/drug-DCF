#!/usr/bin/env python3
"""Build Round 13 Step 2 prototype-response feature ablation manifests."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_runner import _expand_finetune_combinations
from tools.round9_diagnostics_common import load_json, resolve_path

ROUND12_BEST = 0.5971789386885913
ROUND11_BEST = 0.5828

FEATURE_MODE_DEFAULTS = {
    "none": {
        "prototype_distance_metric": "cosine",
        "include_l2_distance": False,
        "include_same_cancer_gap": True,
        "include_initialized_flag": True,
        "proto_feature_scaler": "none",
    },
    "own_cancer": {
        "prototype_distance_metric": "cosine",
        "include_l2_distance": True,
        "include_same_cancer_gap": True,
        "include_initialized_flag": True,
        "proto_feature_scaler": "standard",
    },
    "all_source_anchors": {
        "prototype_distance_metric": "cosine",
        "include_l2_distance": False,
        "include_same_cancer_gap": True,
        "include_initialized_flag": True,
        "proto_feature_scaler": "standard",
    },
    "all_source_and_target": {
        "prototype_distance_metric": "cosine",
        "include_l2_distance": False,
        "include_same_cancer_gap": True,
        "include_initialized_flag": True,
        "proto_feature_scaler": "standard",
    },
    "own_plus_summary": {
        "prototype_distance_metric": "cosine",
        "include_l2_distance": True,
        "include_same_cancer_gap": True,
        "include_initialized_flag": True,
        "proto_feature_scaler": "standard",
    },
}


def _parse_round12_best_from_docs(docs_path: str) -> Optional[str]:
    path = resolve_path(docs_path)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"Best model[:\s`*]+(exp_\d+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _load_round12_pretrain_summary(round12_root: str) -> pd.DataFrame:
    candidates = [
        os.path.join(round12_root, "reports", "round12_proto_pretrain_summary.csv"),
        os.path.join(round12_root, "final_report", "round12_proto_pretrain_summary.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return pd.read_csv(path)
    rows = []
    pretrain_dir = os.path.join(round12_root, "pretrain")
    if os.path.isdir(pretrain_dir):
        for exp_dir in sorted(os.listdir(pretrain_dir)):
            summary_path = os.path.join(pretrain_dir, exp_dir, "run_summary.json")
            if not os.path.isfile(summary_path):
                continue
            payload = load_json(summary_path)
            row = {"model_id": exp_dir, **payload.get("metrics", {})}
            row.update(payload.get("params", {}))
            rows.append(row)
    return pd.DataFrame(rows)


def _resolve_checkpoint(round_root: str, model_id: str) -> str:
    path = os.path.join(resolve_path(round_root), "pretrain", model_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def resolve_model_pool(settings: dict) -> Tuple[List[Dict[str, Any]], List[str]]:
    round12_root = resolve_path(settings["round12_root"])
    round11_root = resolve_path(settings.get("round11_root", "result/optimization_runs/round11_stability_recon"))
    pool_cfg = settings.get("model_pool", {})
    max_models = int(pool_cfg.get("max_models", 8))
    warnings_out: List[str] = []

    primary = str(pool_cfg.get("primary_best_model", "exp_037"))
    docs_best = _parse_round12_best_from_docs(settings.get("round12_final_report", "docs/round12_final_report.md"))
    if docs_best and docs_best != primary:
        warnings_out.append(f"docs best {docs_best} differs from settings primary {primary}; using settings primary")

    aggregate_path = os.path.join(round12_root, "aggregate", "aggregate_scores.csv")
    aggregate = pd.read_csv(aggregate_path) if os.path.isfile(aggregate_path) else pd.DataFrame()
    pretrain_summary = _load_round12_pretrain_summary(round12_root)

    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _add(model_id: str, role: str, source_round: str, checkpoint_dir: str, extra: Optional[dict] = None) -> None:
        if model_id in seen:
            return
        seen.add(model_id)
        row = {
            "source_model_id": model_id,
            "role": role,
            "source_round": source_round,
            "checkpoint_dir": checkpoint_dir,
        }
        if extra:
            row.update(extra)
        selected.append(row)

    _add(primary, "round12_best_downstream", "round12", _resolve_checkpoint(round12_root, primary))

    if pool_cfg.get("include_round11_exp035", True):
        try:
            _add("exp_035", "round11_reference", "round11", _resolve_checkpoint(round11_root, "exp_035"))
        except FileNotFoundError:
            warnings_out.append("Round 11 exp_035 checkpoint missing; skipped")

    if not pretrain_summary.empty:
        df = pretrain_summary.copy()
        if "source_anchor_proto_enabled" in df.columns:
            no_proto = df[df["source_anchor_proto_enabled"].astype(str).str.lower().isin(["false", "0"])]
            if not no_proto.empty and "mean_target_to_source_anchor_distance" in no_proto.columns:
                best_no_proto = no_proto.sort_values("mean_target_to_source_anchor_distance").iloc[0]
                mid = str(best_no_proto["model_id"])
                try:
                    _add(mid, "round12_best_no_proto_control", "round12", _resolve_checkpoint(round12_root, mid))
                except FileNotFoundError:
                    warnings_out.append(f"no-proto control {mid} checkpoint missing")

            proto_on = df[df["source_anchor_proto_enabled"].astype(str).str.lower().isin(["true", "1"])]
            if not proto_on.empty and "mean_target_to_source_anchor_distance" in proto_on.columns:
                best_gap = proto_on.sort_values("mean_target_to_source_anchor_distance").iloc[0]
                mid = str(best_gap["model_id"])
                if mid != primary:
                    try:
                        _add(
                            mid,
                            "round12_best_proto_gap_reduction",
                            "round12",
                            _resolve_checkpoint(round12_root, mid),
                        )
                    except FileNotFoundError:
                        warnings_out.append(f"proto-gap model {mid} checkpoint missing")

            hybrid = proto_on[
                proto_on.get("reconstruction_loss_type", pd.Series("", index=proto_on.index))
                .astype(str)
                .str.contains("smooth|hybrid", case=False, regex=True, na=False)
            ]
            if not hybrid.empty:
                best_h = hybrid.sort_values("mean_target_to_source_anchor_distance").iloc[0]
                mid = str(best_h["model_id"])
                if mid not in seen:
                    try:
                        _add(
                            mid,
                            "round12_hybrid_smoothl1_proto",
                            "round12",
                            _resolve_checkpoint(round12_root, mid),
                        )
                    except FileNotFoundError:
                        warnings_out.append(f"hybrid/smoothl1 model {mid} checkpoint missing")

            if "mean_conditional_leakage_strength" in proto_on.columns:
                leak_safe = proto_on.sort_values("mean_conditional_leakage_strength").iloc[0]
                mid = str(leak_safe["model_id"])
                if mid not in seen:
                    try:
                        _add(
                            mid,
                            "round12_best_leakage_safe",
                            "round12",
                            _resolve_checkpoint(round12_root, mid),
                        )
                    except FileNotFoundError:
                        warnings_out.append(f"leakage-safe model {mid} checkpoint missing")

    if not aggregate.empty and "Average_TCGA_AUC_mean" in aggregate.columns:
        id_col = "Model_ID" if "Model_ID" in aggregate.columns else "model_id"
        top = aggregate.sort_values("Average_TCGA_AUC_mean", ascending=False).head(3)
        for _, row in top.iterrows():
            mid = str(row[id_col])
            if mid in seen:
                continue
            try:
                _add(mid, "round12_aggregate_top", "round12", _resolve_checkpoint(round12_root, mid))
            except FileNotFoundError:
                continue

    if len(selected) < 2:
        warnings_out.append("Model pool smaller than expected; at minimum exp_037 + exp_035 should exist")

    return selected[:max_models], warnings_out


def build_round13_configs(settings_path: str, outdir: str, force: bool = False) -> Dict[str, str]:
    settings = load_json(settings_path)
    outdir = resolve_path(outdir)
    manifests_dir = os.path.join(outdir, "manifests")
    configs_dir = os.path.join(outdir, "configs")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    feature_modes = list(settings.get("feature_modes", ["none", "own_cancer", "all_source_anchors", "all_source_and_target"]))
    optional_modes = list(settings.get("optional_feature_modes", []))
    for mode in optional_modes:
        if mode not in feature_modes:
            feature_modes.append(mode)

    model_pool, pool_warnings = resolve_model_pool(settings)
    for msg in pool_warnings:
        warnings.warn(msg)

    pool_df = pd.DataFrame(model_pool)
    pool_path = os.path.join(manifests_dir, "model_pool.csv")
    pool_df.to_csv(pool_path, index=False)

    proto_rows = []
    finetune_rows = []
    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_proto_features.json")
    combos = _expand_finetune_combinations(finetune_config)

    for model in model_pool:
        source_id = model["source_model_id"]
        for feature_mode in feature_modes:
            defaults = FEATURE_MODE_DEFAULTS.get(feature_mode, FEATURE_MODE_DEFAULTS["own_cancer"])
            job_key = f"r13_{source_id}_{feature_mode}"
            feature_dir = os.path.join(outdir, "features", source_id, feature_mode)
            proto_job_id = f"feat_{source_id}_{feature_mode}"
            proto_rows.append(
                {
                    "job_id": proto_job_id,
                    "source_model_id": source_id,
                    "source_round": model["source_round"],
                    "checkpoint_dir": model["checkpoint_dir"],
                    "role": model.get("role", ""),
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
                "result_folder": feature_dir if feature_mode != "none" else model["checkpoint_dir"],
                "selection_rank": 1,
                "prototype_feature_mode": feature_mode,
                "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                "source_model_id": source_id,
                "source_round": model["source_round"],
            }
            pd.DataFrame([ms_row]).to_csv(model_select_path, index=False)

            for combo in combos:
                ft_job_id = f"ft_{job_key}_c{combo['combo_id']:02d}"
                finetune_rows.append(
                    {
                        "job_id": ft_job_id,
                        "model_id": job_key,
                        "source_model_id": source_id,
                        "source_round": model["source_round"],
                        "feature_mode": feature_mode,
                        "prototype_feature_mode": feature_mode,
                        "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
                        "combined_latent_dir": feature_dir if feature_mode != "none" else model["checkpoint_dir"],
                        "pretrain_result_dir": model["checkpoint_dir"],
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

    proto_manifest = os.path.join(manifests_dir, "proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)

    meta = {
        "round": "round13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models": len(model_pool),
        "n_feature_modes": len(feature_modes),
        "n_finetune_jobs": len(finetune_rows),
        "feature_modes": feature_modes,
        "pool_warnings": pool_warnings,
        "references": settings.get("references", {}),
    }
    with open(os.path.join(outdir, "round13_build_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote model pool ({len(model_pool)} models) -> {pool_path}")
    print(f"Wrote proto feature manifest ({len(proto_rows)} rows) -> {proto_manifest}")
    print(f"Wrote finetune manifest ({len(finetune_rows)} jobs) -> {finetune_manifest}")
    return {
        "model_pool": pool_path,
        "proto_feature_manifest": proto_manifest,
        "finetune_dispatch_manifest": finetune_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 13 prototype response manifests")
    parser.add_argument("--settings", default="config/round13_proto_response_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round13_proto_response")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_round13_configs(args.settings, args.outdir, force=args.force)


if __name__ == "__main__":
    main()
