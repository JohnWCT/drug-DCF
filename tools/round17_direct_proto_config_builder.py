#!/usr/bin/env python3
"""Build Round 17 direct-prototype optimization manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round16_bruteforce_config_builder import (
    ROUND16_MODEL_SPECS,
    _append_finetune_jobs,
    _load_combos,
    _feature_defaults,
)
from tools.round9_diagnostics_common import load_json, resolve_path

ROUND17_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    **ROUND16_MODEL_SPECS,
    # Explicit control aliases (not auto-resolved Round 16 tops). See settings model_aliases.
    "r13_exp_035_control": ROUND16_MODEL_SPECS["r13_exp_035"],
    "r13_exp_008_control": ROUND16_MODEL_SPECS["r13_exp_008"],
    # Deprecated misleading names — kept for backward compatibility only.
    "round16_top1": ROUND16_MODEL_SPECS["r13_exp_035"],
    "round16_top2": ROUND16_MODEL_SPECS["r13_exp_008"],
}


def _resolve_round17_models(settings: dict, model_keys: List[str]) -> List[str]:
    resolved = []
    for key in model_keys:
        if key in ROUND17_MODEL_SPECS:
            resolved.append(key)
        else:
            raise ValueError(f"Unknown Round 17 model key: {key}")
    return resolved


def _build_model_pool_round17(settings: dict, model_keys: List[str]) -> pd.DataFrame:
    rows = []
    for model_key in _resolve_round17_models(settings, model_keys):
        spec = ROUND17_MODEL_SPECS[model_key]
        root = resolve_path(settings[spec["checkpoint_root_key"]])
        checkpoint_dir = os.path.join(root, "pretrain", spec["checkpoint_subdir"])
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found for {model_key}: {checkpoint_dir}")
        rows.append(
            {
                "model_id": model_key,
                "source_model_id": spec["source_model_id"],
                "source_round": spec["source_round"],
                "checkpoint_dir": checkpoint_dir,
            }
        )
    return pd.DataFrame(rows)


def _resolve_checkpoint_round17(settings: dict, model_key: str):
    spec = ROUND17_MODEL_SPECS[model_key]
    root = resolve_path(settings[spec["checkpoint_root_key"]])
    checkpoint_dir = os.path.join(root, "pretrain", spec["checkpoint_subdir"])
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found for {model_key}: {checkpoint_dir}")
    return checkpoint_dir, spec


def _stage_enabled(settings: dict, stage_key: str) -> bool:
    return bool(settings.get(stage_key, {}).get("enabled", False))


def build_stage17a(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings["stage17a"]
    if not stage_cfg.get("enabled", True):
        raise ValueError("stage17a is disabled in settings")

    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    model_pool = _build_model_pool_round17(settings, list(stage_cfg["models"]))
    combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 8))
    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []

    _append_finetune_jobs(
        settings=settings,
        outdir=outdir,
        stage="17a",
        model_pool=model_pool,
        feature_modes=list(stage_cfg["feature_modes"]),
        combos=combos,
        seeds=list(stage_cfg["seeds"]),
        finetune_rows=finetune_rows,
        proto_rows=proto_rows,
        model_select_dir=model_select_dir,
        feature_root=os.path.join(outdir, "features"),
        finetune_root=os.path.join(outdir, "stage17a", "finetune"),
    )

    pool_path = os.path.join(manifests_dir, "model_pool.csv")
    model_pool.to_csv(pool_path, index=False)
    proto_manifest = os.path.join(manifests_dir, "stage17a_proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "stage17a_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)

    return {
        "model_pool": pool_path,
        "stage17a_proto_feature_manifest": proto_manifest,
        "stage17a_finetune_dispatch_manifest": finetune_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage17b(settings: dict, outdir: str, top_candidates_path: str) -> Dict[str, str]:
    stage_cfg = settings["stage17b"]
    if not stage_cfg.get("enabled", True):
        raise ValueError("stage17b is disabled in settings")

    candidates = pd.read_csv(resolve_path(top_candidates_path))
    if candidates.empty:
        raise ValueError(f"No candidates in {top_candidates_path}")

    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects_stage17b")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(model_select_dir, exist_ok=True)

    finetune_rows: List[dict] = []
    proto_rows: List[dict] = []
    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round17_direct_proto.json")
    all_combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 8))
    heads = list(stage_cfg.get("heads", ["concat_mlp"]))

    for _, cand in candidates.head(int(stage_cfg.get("top_k_feature_candidates", 10))).iterrows():
        model_key = str(cand.get("round17_model_key", cand.get("model_key", "")))
        if model_key not in ROUND17_MODEL_SPECS:
            for key in ROUND17_MODEL_SPECS:
                if key in str(cand.get("model_id", "")):
                    model_key = key
                    break
        feature_mode = str(cand.get("feature_mode", "none"))
        combo_id = int(cand.get("combo_id", 0))
        checkpoint_dir, spec = _resolve_checkpoint_round17(settings, model_key)
        job_key = f"{model_key}_{feature_mode}"
        feature_dir = os.path.join(outdir, "features", model_key, feature_mode)
        defaults = _feature_defaults(feature_mode)

        for head_mode in heads:
            proto_job_id = f"feat_{job_key}_{head_mode}"
            proto_rows.append(
                {
                    "job_id": proto_job_id,
                    "stage": "17b",
                    "model_id": model_key,
                    "source_model_id": spec["source_model_id"],
                    "source_round": spec["source_round"],
                    "checkpoint_dir": checkpoint_dir,
                    "feature_mode": feature_mode,
                    "response_head_mode": head_mode,
                    "combined_latent_dir": feature_dir,
                    "status": "pending",
                    **{k: defaults[k] for k in defaults if k.startswith("include_") or k.endswith("_metric") or k == "proto_feature_scaler"},
                }
            )
            src_ms = os.path.join(outdir, "manifests", "model_selects", f"{job_key}.csv")
            model_select_path = src_ms if os.path.isfile(src_ms) else os.path.join(model_select_dir, f"{job_key}.csv")
            combo_row = all_combos[combo_id] if combo_id < len(all_combos) else all_combos[0]
            for seed in stage_cfg["seeds"]:
                ft_job_id = f"ft_17b_{job_key}_{head_mode}_c{combo_id:02d}_s{seed}"
                finetune_rows.append(
                    {
                        "job_id": ft_job_id,
                        "stage": "17b",
                        "model_id": job_key,
                        "source_model_id": spec["source_model_id"],
                        "source_round": spec["source_round"],
                        "feature_mode": feature_mode,
                        "response_head_mode": head_mode,
                        "combined_latent_dir": feature_dir,
                        "pretrain_result_dir": checkpoint_dir,
                        "model_select_path": model_select_path,
                        "finetune_config_path": finetune_config,
                        "combo_id": combo_id,
                        "seed": seed,
                        "batch_size": combo_row.get("batch_size", settings.get("finetune", {}).get("batch_size", 12288)),
                        "mini_batch_size": combo_row.get(
                            "mini_batch_size", settings.get("finetune", {}).get("mini_batch_size", 3072)
                        ),
                        "epochs": combo_row.get("epochs", settings.get("finetune", {}).get("epochs", 1500)),
                        "result_dir": os.path.join(
                            outdir, "stage17b", "finetune", job_key, head_mode, f"combo_{combo_id:02d}", f"seed_{seed}"
                        ),
                        "status": "pending",
                        "start_time": "",
                        "end_time": "",
                        "error_message": "",
                    }
                )

    proto_manifest = os.path.join(manifests_dir, "stage17b_proto_feature_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_manifest, index=False)
    finetune_manifest = os.path.join(manifests_dir, "stage17b_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)
    return {
        "stage17b_proto_feature_manifest": proto_manifest,
        "stage17b_finetune_dispatch_manifest": finetune_manifest,
        "n_jobs": len(finetune_rows),
    }


def build_stage17c(settings: dict, outdir: str, top_candidates_path: str) -> Dict[str, str]:
    stage_cfg = settings["stage17c"]
    if not stage_cfg.get("enabled", True):
        raise ValueError("stage17c is disabled in settings")

    candidates = pd.read_csv(resolve_path(top_candidates_path)).head(int(stage_cfg.get("top_k_candidates", 5)))
    manifests_dir = os.path.join(outdir, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)
    finetune_rows: List[dict] = []
    finetune_config = settings.get("finetune", {}).get("config", "config/params_finetune_round17_direct_proto.json")

    for _, cand in candidates.iterrows():
        model_key = str(cand.get("round17_model_key", cand.get("model_key", "")))
        feature_mode = str(cand.get("feature_mode", "none"))
        head_mode = str(cand.get("response_head_mode", "concat_mlp"))
        combo_id = int(cand.get("combo_id", 0))
        _, spec = _resolve_checkpoint_round17(settings, model_key)
        job_key = f"{model_key}_{feature_mode}"
        feature_dir = os.path.join(outdir, "features", model_key, feature_mode)
        checkpoint_dir, _ = _resolve_checkpoint_round17(settings, model_key)
        model_select_path = os.path.join(outdir, "manifests", "model_selects", f"{job_key}.csv")
        for seed in stage_cfg["seeds"]:
            finetune_rows.append(
                {
                    "job_id": f"ft_17c_{job_key}_{head_mode}_c{combo_id:02d}_s{seed}",
                    "stage": "17c",
                    "model_id": job_key,
                    "source_model_id": spec["source_model_id"],
                    "source_round": spec["source_round"],
                    "feature_mode": feature_mode,
                    "response_head_mode": head_mode,
                    "combined_latent_dir": feature_dir,
                    "pretrain_result_dir": checkpoint_dir,
                    "model_select_path": model_select_path,
                    "finetune_config_path": finetune_config,
                    "combo_id": combo_id,
                    "seed": seed,
                    "result_dir": os.path.join(
                        outdir, "stage17c", "finetune", job_key, head_mode, f"combo_{combo_id:02d}", f"seed_{seed}"
                    ),
                    "status": "pending",
                    "start_time": "",
                    "end_time": "",
                    "error_message": "",
                }
            )

    finetune_manifest = os.path.join(manifests_dir, "stage17c_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(finetune_manifest, index=False)
    return {"stage17c_finetune_dispatch_manifest": finetune_manifest, "n_jobs": len(finetune_rows)}


def build_round17_configs(
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

    if stage == "17a":
        if not _stage_enabled(settings, "stage17a"):
            raise ValueError("stage17a is disabled")
        outputs = build_stage17a(settings, outdir)
    elif stage == "17b":
        if not _stage_enabled(settings, "stage17b"):
            raise ValueError("stage17b is disabled")
        if not top_candidates:
            raise ValueError("--top-candidates required for stage 17b")
        outputs = build_stage17b(settings, outdir, top_candidates)
    elif stage == "17c":
        if not _stage_enabled(settings, "stage17c"):
            raise ValueError("stage17c is disabled")
        if not top_candidates:
            raise ValueError("--top-candidates required for stage 17c")
        outputs = build_stage17c(settings, outdir, top_candidates)
    elif stage == "17f":
        outputs = {"stage": "17f", "note": "tSNE uses visualize_round17_prototype_tsne.py directly"}
    else:
        raise ValueError(f"Unknown stage: {stage}")

    meta = {
        "round": "round17",
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    meta_path = os.path.join(outdir, f"round17_build_{stage}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Round 17 stage {stage}: wrote {outputs.get('n_jobs', '?')} jobs -> {outdir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 17 direct-prototype manifests")
    parser.add_argument("--settings", default="config/round17_direct_proto_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round17_direct_proto")
    parser.add_argument("--stage", required=True, choices=["17a", "17b", "17c", "17f"])
    parser.add_argument("--top-candidates", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_round17_configs(args.settings, args.outdir, args.stage, args.top_candidates, force=args.force)


if __name__ == "__main__":
    main()
