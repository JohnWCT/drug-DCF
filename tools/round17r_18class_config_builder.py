#!/usr/bin/env python3
"""Build Round 17R 18-class-clean focused manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_runner import _expand_finetune_combinations
from tools.round13_config_builder import FEATURE_MODE_DEFAULTS
from tools.round16_bruteforce_config_builder import ROUND16_MODEL_SPECS, _feature_defaults
from tools.round9_diagnostics_common import load_json, resolve_path
from tools.prototype_response_features import resolve_feature_mode_options

ROUND17R_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    **ROUND16_MODEL_SPECS,
    "r13_exp_035_control": ROUND16_MODEL_SPECS["r13_exp_035"],
    "r13_exp_008_control": ROUND16_MODEL_SPECS["r13_exp_008"],
}

FORBIDDEN_PLACEHOLDERS = ("round16_top", "round16_top1", "round16_top2")


def _reject_forbidden_aliases(model_key: str) -> None:
    key = str(model_key).strip().lower()
    for bad in FORBIDDEN_PLACEHOLDERS:
        if key == bad or key.startswith(bad):
            raise ValueError(
                f"Forbidden Round 17R model alias {model_key!r}; "
                "do not auto-resolve round16_top placeholders"
            )


def _resolve_checkpoint(settings: dict, model_key: str) -> Tuple[str, dict]:
    _reject_forbidden_aliases(model_key)
    if model_key not in ROUND17R_MODEL_SPECS:
        raise ValueError(f"Unknown Round 17R model key: {model_key}")
    spec = ROUND17R_MODEL_SPECS[model_key]
    root = resolve_path(settings[spec["checkpoint_root_key"]])
    checkpoint_dir = os.path.join(root, "pretrain", spec["checkpoint_subdir"])
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found for {model_key}: {checkpoint_dir}")
    return checkpoint_dir, spec


def _load_combos(settings: dict, max_combos: Optional[int] = None) -> List[dict]:
    finetune_cfg = settings.get("finetune", {}).get("config") or settings.get(
        "finetune_config", "config/params_finetune_round17r_focused.json"
    )
    combos = _expand_finetune_combinations(finetune_cfg)
    if max_combos is not None:
        combos = [c for c in combos if int(c["combo_id"]) < int(max_combos)]
    return combos


def _require_n(settings: dict) -> int:
    return int(settings.get("require_n_trainable_cancer_types", 18))


def _drug_smiles_path(settings: dict) -> str:
    return str(
        settings.get("drug_smiles_path")
        or settings.get("finetune", {}).get("drug_smiles_path")
        or "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"
    )


def _finetune_config_path(settings: dict) -> str:
    return str(
        settings.get("finetune", {}).get("config")
        or settings.get("finetune_config")
        or "config/params_finetune_round17r_focused.json"
    )


def _write_model_select(
    path: str,
    *,
    job_key: str,
    checkpoint_dir: str,
    feature_dir: str,
    feature_mode: str,
    source_id: str,
    source_round: str,
    model_key: str,
) -> None:
    opts = resolve_feature_mode_options(feature_mode)
    ms_row = {
        "ID": job_key,
        "model_type": "VAE",
        "result_folder": checkpoint_dir if feature_mode == "none" else feature_dir,
        "selection_rank": 1,
        "prototype_feature_mode": opts["feature_mode_label"],
        "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
        "source_model_id": source_id,
        "source_round": source_round,
        "round17r_model_key": model_key,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([ms_row]).to_csv(path, index=False)


def _proto_row(
    *,
    settings: dict,
    stage: str,
    model_key: str,
    feature_mode: str,
    checkpoint_dir: str,
    spec: dict,
    feature_dir: str,
) -> dict:
    defaults = _feature_defaults(feature_mode)
    opts = resolve_feature_mode_options(feature_mode)
    return {
        "job_id": f"feat_{stage}_{model_key}_{feature_mode}",
        "stage": stage,
        "model_id": model_key,
        "model_key": model_key,
        "source_model_id": spec["source_model_id"],
        "source_round": spec["source_round"],
        "checkpoint_dir": checkpoint_dir,
        "pretrain_dir": checkpoint_dir,
        "feature_mode": feature_mode,
        "prototype_feature_mode": opts["feature_mode_label"],
        "feature_variant": feature_mode,
        "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
        "prototype_distance_metric": defaults["prototype_distance_metric"],
        "include_l2_distance": defaults["include_l2_distance"],
        "include_same_cancer_gap": defaults["include_same_cancer_gap"],
        "include_initialized_flag": defaults["include_initialized_flag"],
        "proto_feature_scaler": defaults["proto_feature_scaler"],
        "combined_latent_dir": feature_dir,
        "require_n_trainable_cancer_types": _require_n(settings),
        "prototype_class_source": settings.get("prototype_class_source", "checkpoint_metadata"),
        "drug_smiles_path": _drug_smiles_path(settings),
        "status": "pending",
    }


def _finetune_row(
    *,
    settings: dict,
    stage: str,
    model_key: str,
    feature_mode: str,
    checkpoint_dir: str,
    spec: dict,
    feature_dir: str,
    model_select_path: str,
    combo: dict,
    seed: int,
    result_dir: str,
) -> dict:
    opts = resolve_feature_mode_options(feature_mode)
    combo_id = int(combo["combo_id"])
    return {
        "job_id": f"ft_{stage}_{model_key}_{feature_mode}_c{combo_id:02d}_s{seed}",
        "stage": stage,
        "model_id": f"{model_key}_{feature_mode}",
        "model_key": model_key,
        "source_model_id": spec["source_model_id"],
        "source_round": spec["source_round"],
        "feature_mode": feature_mode,
        "prototype_feature_mode": opts["feature_mode_label"],
        "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
        "combined_latent_dir": feature_dir if feature_mode != "none" else checkpoint_dir,
        "pretrain_result_dir": checkpoint_dir,
        "pretrain_dir": checkpoint_dir,
        "model_select_path": model_select_path,
        "finetune_config_path": _finetune_config_path(settings),
        "drug_smiles_path": _drug_smiles_path(settings),
        "combo_id": combo_id,
        "seed": int(seed),
        "random_seed": int(seed),
        "batch_size": int(combo.get("batch_size", settings.get("finetune", {}).get("batch_size", 24576))),
        "mini_batch_size": int(
            combo.get("mini_batch_size", settings.get("finetune", {}).get("mini_batch_size", 6144))
        ),
        "epochs": int(combo.get("epochs", settings.get("finetune", {}).get("epochs", 1500))),
        "result_dir": result_dir,
        "require_n_trainable_cancer_types": _require_n(settings),
        "status": "pending",
        "start_time": "",
        "end_time": "",
        "error_message": "",
    }


def _assert_paths_exist(paths: Sequence[str]) -> None:
    missing = [p for p in paths if p and not os.path.exists(resolve_path(p))]
    if missing:
        raise FileNotFoundError("Missing required paths:\n- " + "\n- ".join(missing))


def build_stage17r_a(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings["stage17r_a"]
    manifests_dir = os.path.join(outdir, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)
    rows: List[dict] = []
    for model_key in stage_cfg["models"]:
        checkpoint_dir, spec = _resolve_checkpoint(settings, model_key)
        for feature_mode in stage_cfg["feature_modes"]:
            feature_dir = os.path.join(outdir, "features", model_key, feature_mode)
            rows.append(
                _proto_row(
                    settings=settings,
                    stage="17r_a",
                    model_key=model_key,
                    feature_mode=feature_mode,
                    checkpoint_dir=checkpoint_dir,
                    spec=spec,
                    feature_dir=feature_dir,
                )
            )
    max_jobs = int(stage_cfg.get("max_jobs", len(rows)))
    rows = rows[:max_jobs]
    path = os.path.join(manifests_dir, "stage17r_a_proto_feature_manifest.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage17r_a_proto_feature_manifest": path, "n_jobs": len(rows)}


def build_stage17r_b(settings: dict, outdir: str) -> Dict[str, str]:
    stage_cfg = settings["stage17r_b"]
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(model_select_dir, exist_ok=True)
    combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 6))
    proto_rows: List[dict] = []
    finetune_rows: List[dict] = []
    seen_feat = set()

    for model_key, feature_mode in stage_cfg["candidates"]:
        _reject_forbidden_aliases(model_key)
        checkpoint_dir, spec = _resolve_checkpoint(settings, model_key)
        job_key = f"{model_key}_{feature_mode}"
        feature_dir = os.path.join(outdir, "features", model_key, feature_mode)
        model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
        _write_model_select(
            model_select_path,
            job_key=job_key,
            checkpoint_dir=checkpoint_dir,
            feature_dir=feature_dir,
            feature_mode=feature_mode,
            source_id=spec["source_model_id"],
            source_round=spec["source_round"],
            model_key=model_key,
        )
        feat_key = (model_key, feature_mode)
        if feat_key not in seen_feat:
            proto_rows.append(
                _proto_row(
                    settings=settings,
                    stage="17r_b",
                    model_key=model_key,
                    feature_mode=feature_mode,
                    checkpoint_dir=checkpoint_dir,
                    spec=spec,
                    feature_dir=feature_dir,
                )
            )
            seen_feat.add(feat_key)
        for combo in combos:
            for seed in stage_cfg["seeds"]:
                result_dir = os.path.join(
                    outdir,
                    "stage17r_b",
                    "finetune",
                    job_key,
                    f"combo_{int(combo['combo_id']):02d}",
                    f"seed_{seed}",
                )
                finetune_rows.append(
                    _finetune_row(
                        settings=settings,
                        stage="17r_b",
                        model_key=model_key,
                        feature_mode=feature_mode,
                        checkpoint_dir=checkpoint_dir,
                        spec=spec,
                        feature_dir=feature_dir,
                        model_select_path=model_select_path,
                        combo=combo,
                        seed=seed,
                        result_dir=result_dir,
                    )
                )

    proto_path = os.path.join(manifests_dir, "stage17r_b_proto_feature_manifest.csv")
    ft_path = os.path.join(manifests_dir, "stage17r_b_finetune_dispatch_manifest.csv")
    pd.DataFrame(proto_rows).to_csv(proto_path, index=False)
    pd.DataFrame(finetune_rows).to_csv(ft_path, index=False)
    smiles = resolve_path(_drug_smiles_path(settings))
    _assert_paths_exist([smiles, resolve_path(_finetune_config_path(settings))])
    return {
        "stage17r_b_proto_feature_manifest": proto_path,
        "stage17r_b_finetune_dispatch_manifest": ft_path,
        "n_jobs": len(finetune_rows),
    }


def _candidates_from_top_csv(path: str, top_k: int) -> List[Tuple[str, str]]:
    df = pd.read_csv(resolve_path(path))
    if df.empty:
        raise ValueError(f"No candidates in {path}")
    out: List[Tuple[str, str]] = []
    for _, row in df.head(int(top_k)).iterrows():
        model_key = str(row.get("model_key", row.get("round17r_model_key", row.get("round17_model_key", "")))).strip()
        feature_mode = str(row.get("feature_mode", "")).strip()
        if not model_key or not feature_mode:
            model_id = str(row.get("model_id", ""))
            for key in sorted(ROUND17R_MODEL_SPECS, key=len, reverse=True):
                if model_id.startswith(key + "_"):
                    model_key = key
                    feature_mode = model_id[len(key) + 1 :]
                    break
        if not model_key or not feature_mode:
            raise ValueError(f"Cannot parse candidate row: {dict(row)}")
        _reject_forbidden_aliases(model_key)
        out.append((model_key, feature_mode))
    return out


def build_stage17r_c(settings: dict, outdir: str, top_candidates_path: str) -> Dict[str, str]:
    stage_cfg = settings["stage17r_c"]
    candidates = _candidates_from_top_csv(top_candidates_path, stage_cfg.get("top_k_from_stage17r_b", 6))
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(model_select_dir, exist_ok=True)
    combos = _load_combos(settings, max_combos=stage_cfg.get("max_combos", 6))
    finetune_rows: List[dict] = []
    for model_key, feature_mode in candidates:
        checkpoint_dir, spec = _resolve_checkpoint(settings, model_key)
        job_key = f"{model_key}_{feature_mode}"
        feature_dir = os.path.join(outdir, "features", model_key, feature_mode)
        model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
        if not os.path.isfile(model_select_path):
            _write_model_select(
                model_select_path,
                job_key=job_key,
                checkpoint_dir=checkpoint_dir,
                feature_dir=feature_dir,
                feature_mode=feature_mode,
                source_id=spec["source_model_id"],
                source_round=spec["source_round"],
                model_key=model_key,
            )
        for combo in combos:
            for seed in stage_cfg["seeds"]:
                result_dir = os.path.join(
                    outdir,
                    "stage17r_c",
                    "finetune",
                    job_key,
                    f"combo_{int(combo['combo_id']):02d}",
                    f"seed_{seed}",
                )
                finetune_rows.append(
                    _finetune_row(
                        settings=settings,
                        stage="17r_c",
                        model_key=model_key,
                        feature_mode=feature_mode,
                        checkpoint_dir=checkpoint_dir,
                        spec=spec,
                        feature_dir=feature_dir,
                        model_select_path=model_select_path,
                        combo=combo,
                        seed=seed,
                        result_dir=result_dir,
                    )
                )
    path = os.path.join(manifests_dir, "stage17r_c_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(path, index=False)
    return {"stage17r_c_finetune_dispatch_manifest": path, "n_jobs": len(finetune_rows)}


def build_stage17r_d(settings: dict, outdir: str, top_candidates_path: str) -> Dict[str, str]:
    stage_cfg = settings["stage17r_d"]
    candidates = _candidates_from_top_csv(top_candidates_path, stage_cfg.get("top_k_from_stage17r_c", 5))
    manifests_dir = os.path.join(outdir, "manifests")
    model_select_dir = os.path.join(manifests_dir, "model_selects")
    os.makedirs(model_select_dir, exist_ok=True)
    # confirmation: best combo_id=0 as default (builder uses single combo 0 unless candidate specifies)
    combos = _load_combos(settings, max_combos=1)
    finetune_rows: List[dict] = []
    for model_key, feature_mode in candidates:
        checkpoint_dir, spec = _resolve_checkpoint(settings, model_key)
        job_key = f"{model_key}_{feature_mode}"
        feature_dir = os.path.join(outdir, "features", model_key, feature_mode)
        model_select_path = os.path.join(model_select_dir, f"{job_key}.csv")
        if not os.path.isfile(model_select_path):
            _write_model_select(
                model_select_path,
                job_key=job_key,
                checkpoint_dir=checkpoint_dir,
                feature_dir=feature_dir,
                feature_mode=feature_mode,
                source_id=spec["source_model_id"],
                source_round=spec["source_round"],
                model_key=model_key,
            )
        for combo in combos:
            for seed in stage_cfg["seeds"]:
                result_dir = os.path.join(
                    outdir,
                    "stage17r_d",
                    "finetune",
                    job_key,
                    f"combo_{int(combo['combo_id']):02d}",
                    f"seed_{seed}",
                )
                finetune_rows.append(
                    _finetune_row(
                        settings=settings,
                        stage="17r_d",
                        model_key=model_key,
                        feature_mode=feature_mode,
                        checkpoint_dir=checkpoint_dir,
                        spec=spec,
                        feature_dir=feature_dir,
                        model_select_path=model_select_path,
                        combo=combo,
                        seed=seed,
                        result_dir=result_dir,
                    )
                )
    path = os.path.join(manifests_dir, "stage17r_d_finetune_dispatch_manifest.csv")
    pd.DataFrame(finetune_rows).to_csv(path, index=False)
    return {"stage17r_d_finetune_dispatch_manifest": path, "n_jobs": len(finetune_rows)}


def build_round17r_configs(
    settings_path: str,
    outdir: str,
    stage: str,
    top_candidates: Optional[str] = None,
) -> Dict[str, Any]:
    settings = load_json(settings_path)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    stage = stage.lower().replace("stage", "").strip()

    if stage in ("17r_a", "a"):
        outputs = build_stage17r_a(settings, outdir)
    elif stage in ("17r_b", "b"):
        outputs = build_stage17r_b(settings, outdir)
    elif stage in ("17r_c", "c"):
        if not top_candidates:
            raise ValueError("--top-candidates required for stage 17r_c")
        outputs = build_stage17r_c(settings, outdir, top_candidates)
    elif stage in ("17r_d", "d"):
        if not top_candidates:
            raise ValueError("--top-candidates required for stage 17r_d")
        outputs = build_stage17r_d(settings, outdir, top_candidates)
    elif stage in ("17r_f", "f"):
        outputs = {"stage": "17r_f", "note": "tSNE uses visualize_round17_prototype_tsne.py"}
    else:
        raise ValueError(f"Unknown Round 17R stage: {stage}")

    meta = {
        "round": "round17r",
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    meta_path = os.path.join(outdir, f"round17r_build_{stage}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Round 17R stage {stage}: wrote {outputs.get('n_jobs', '?')} jobs -> {outdir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 17R 18-class-clean manifests")
    parser.add_argument("--settings", default="config/round17r_18class_focused_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round17r_18class")
    parser.add_argument("--stage", required=True, choices=["17r_a", "17r_b", "17r_c", "17r_d", "17r_f", "a", "b", "c", "d", "f"])
    parser.add_argument("--top-candidates", default=None)
    args = parser.parse_args()
    build_round17r_configs(args.settings, args.outdir, args.stage, args.top_candidates)


if __name__ == "__main__":
    main()
