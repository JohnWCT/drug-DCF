"""Round 18 config / manifest builder."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_splits import write_round18_splits
from tools.round18_eligible_data import build_round18_eligible_response


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_dir_for_omics(settings: dict, omics_mode: str) -> str:
    root = settings.get("feature_root", "result/optimization_runs/round17r_18class/features")
    model_key = settings.get("feature_model_key", "r13_exp_008")
    mode = "none" if omics_mode in {"none", "z-only", "z_only"} else omics_mode
    return str(Path(root) / model_key / mode)


def _validate_feature_dirs(settings: dict) -> None:
    from tools.round18_eligible_data import validate_feature_metadata

    for omics in settings.get("omics_modes", []):
        fdir = feature_dir_for_omics(settings, omics)
        validate_feature_metadata(fdir)


def build_stage18a(settings: dict, outdir: str) -> Dict[str, Any]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    _validate_feature_dirs(settings)

    # Eligible table uses primary omics feature set (own_plus_summary)
    primary_omics = "own_plus_summary"
    if primary_omics not in settings.get("omics_modes", []):
        primary_omics = settings.get("omics_modes", ["own_plus_summary"])[0]
    feature_dir = feature_dir_for_omics(settings, primary_omics)

    eligible_summary = build_round18_eligible_response(
        settings["response_data_path"],
        feature_dir=feature_dir,
        drug_smiles_path=settings["drug_smiles_path"],
        outdir=outdir,
        group_column=settings["internal_test"]["group_column"],
        label_column=settings["internal_test"]["label_column"],
    )
    eligible_path = eligible_summary["paths"]["eligible"]

    paths = write_round18_splits(
        eligible_path,
        outdir,
        group_column=settings["internal_test"]["group_column"],
        label_column=settings["internal_test"]["label_column"],
        drug_column="DRUG_NAME",
        split_seed=int(settings.get("split_seed", 42)),
        screening_folds=int(settings["screening_cv"]["n_splits"]),
        formal_folds=int(settings["formal_cv"]["n_splits"]),
        require_eligible=True,
    )
    meta = {
        "stage": "18a",
        "outdir": str(out),
        "eligible_summary": eligible_summary,
        "split_artifacts": paths,
        "settings_snapshot": {
            "model_seed": settings.get("model_seed"),
            "split_seed": settings.get("split_seed"),
            "omics_modes": settings.get("omics_modes"),
            "gin": settings.get("gin"),
        },
    }
    meta_path = out / "round18_build_18a_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"stage": "18a", "metadata": str(meta_path), "eligible": eligible_summary, **paths}


def _base_job(
    *,
    job_id: str,
    stage: str,
    architecture_id: str,
    architecture_family: str,
    omics_mode: str,
    settings: dict,
    outdir: str,
    fold_id: int,
    cv_type: str,
    transformer_config_id: str = "",
    residual_mode: str = "",
    global_lr: Optional[float] = None,
    mask_meta: Optional[dict] = None,
) -> dict:
    row = {
        "job_id": job_id,
        "stage": stage,
        "architecture_id": architecture_id,
        "architecture_family": architecture_family,
        "omics_mode": omics_mode,
        "transformer_config_id": transformer_config_id,
        "residual_mode": residual_mode,
        "cv_type": cv_type,
        "fold_id": fold_id,
        "model_seed": settings.get("model_seed", 101),
        "split_seed": settings.get("split_seed", 42),
        "drug_smiles_path": settings["drug_smiles_path"],
        "response_data_path": str(Path(outdir) / "data" / "round18_eligible_response.csv"),
        "feature_dir": feature_dir_for_omics(settings, omics_mode),
        "split_assignment": str(
            Path(outdir)
            / "splits"
            / ("screening_3fold_assignments.csv" if "screening" in cv_type else "formal_5fold_assignments.csv")
        ),
        "result_dir": str(Path(outdir) / f"stage{stage}" / architecture_id / f"fold_{fold_id}"),
        "requested_micro_batch": settings["oom"]["micro_batch_candidates"][0],
        "target_effective_batch": settings["oom"]["target_effective_batch"],
        "status": "pending",
    }
    if global_lr is not None:
        row["global_lr"] = global_lr
    if mask_meta:
        row.update(mask_meta)
    return row


def build_stage18b_manifest(settings: dict, screening: dict, outdir: str) -> Dict[str, Any]:
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    omics_modes = settings.get("omics_modes", ["none", "own_plus_summary", "own_proto_context_projected_16"])
    n_folds = int(settings["screening_cv"]["n_splits"])
    rows: List[dict] = []

    for omics in omics_modes:
        for fold_id in range(n_folds):
            rows.append(
                _base_job(
                    job_id=f"18b_mlp_{omics}_f{fold_id}",
                    stage="18b",
                    architecture_id=f"pooled_mlp__{omics}",
                    architecture_family="pooled_mlp",
                    omics_mode=omics,
                    settings=settings,
                    outdir=outdir,
                    fold_id=fold_id,
                    cv_type="screening_3fold",
                )
            )

    for cfg in screening.get("pooled_transformer_configs", []):
        cfg_id = cfg["config_id"]
        # rename historical for corrected mask semantics
        display_id = cfg_id
        if cfg_id == "P0_historical":
            display_id = "P0_historical_hparams_corrected_mask"
        mask_meta = {
            "requested_use_mask": bool(cfg.get("use_mask", True)),
            "effective_use_mask": False,
            "mask_reason": "two dense tokens without padding",
        }
        for omics in omics_modes:
            for fold_id in range(n_folds):
                rows.append(
                    _base_job(
                        job_id=f"18b_tf_{display_id}_{omics}_f{fold_id}",
                        stage="18b",
                        architecture_id=f"pooled_transformer__{display_id}__{omics}",
                        architecture_family="pooled_transformer",
                        omics_mode=omics,
                        settings=settings,
                        outdir=outdir,
                        fold_id=fold_id,
                        cv_type="screening_3fold",
                        transformer_config_id=display_id,
                        global_lr=float(cfg["global_lr"]) if "global_lr" in cfg else None,
                        mask_meta=mask_meta,
                    )
                )

    path = manifests / "stage18b_screening_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage": "18b", "manifest": str(path), "n_jobs": len(rows)}


def build_stage18c_manifest(settings: dict, screening: dict, outdir: str) -> Dict[str, Any]:
    """
    Stage 18C-A: cross-attention × residual × omics × 3fold.

    Omics include own_plus_summary and context16 (not none in the main 48-job grid).
    Optional follow-up: top-2 × none × 3fold = +6 jobs (built separately).
    """
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    n_folds = int(settings["screening_cv"]["n_splits"])
    cross_attention_omics_modes = [
        "own_plus_summary",
        "own_proto_context_projected_16",
    ]
    rows: List[dict] = []
    for cfg in screening.get("cross_attention_configs", []):
        cfg_id = cfg["config_id"]
        for residual_mode in ("pure", "pooled_residual"):
            for omics_mode in cross_attention_omics_modes:
                for fold_id in range(n_folds):
                    arch = f"cross_attn__{cfg_id}__{residual_mode}__{omics_mode}"
                    rows.append(
                        _base_job(
                            job_id=f"18c_{cfg_id}_{residual_mode}_{omics_mode}_f{fold_id}",
                            stage="18c",
                            architecture_id=arch,
                            architecture_family="cross_attention",
                            omics_mode=omics_mode,
                            settings=settings,
                            outdir=outdir,
                            fold_id=fold_id,
                            cv_type="screening_3fold",
                            transformer_config_id=cfg_id,
                            residual_mode=residual_mode,
                        )
                    )
    path = manifests / "stage18c_cross_attention_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {
        "stage": "18c",
        "manifest": str(path),
        "n_jobs": len(rows),
        "omics_modes": cross_attention_omics_modes,
        "note": "18C-A grid excludes none; add top2×none×3fold after ranking",
    }


def build_stage18c_none_followup_manifest(
    settings: dict,
    outdir: str,
    *,
    top_candidates: Optional[List[dict]] = None,
    locked_selection_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    18C-B follow-up: top cross-attention candidates × none × 3 folds.

    top_candidates entries need architecture_family=cross_attention fields:
      architecture_id, transformer_config_id, residual_mode
    """
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if top_candidates is None:
        lock_path = Path(locked_selection_path or Path(outdir) / "reports" / "round18_18c_top_for_none.json")
        if not lock_path.is_file():
            raise FileNotFoundError(
                f"Missing {lock_path}; provide top_candidates or write 18C top-for-none file first"
            )
        top_candidates = load_json(str(lock_path)).get("top_cross_attention_for_none", [])
    if len(top_candidates) < 1:
        raise ValueError("Need at least one top cross-attention candidate for none follow-up")

    n_folds = int(settings["screening_cv"]["n_splits"])
    rows: List[dict] = []
    for cand in top_candidates[:2]:
        cfg_id = cand.get("transformer_config_id") or ""
        residual_mode = cand.get("residual_mode") or "pure"
        for fold_id in range(n_folds):
            arch = f"cross_attn__{cfg_id}__{residual_mode}__none"
            rows.append(
                _base_job(
                    job_id=f"18c_{cfg_id}_{residual_mode}_none_f{fold_id}",
                    stage="18c",
                    architecture_id=arch,
                    architecture_family="cross_attention",
                    omics_mode="none",
                    settings=settings,
                    outdir=outdir,
                    fold_id=fold_id,
                    cv_type="screening_3fold",
                    transformer_config_id=cfg_id,
                    residual_mode=residual_mode,
                )
            )
    path = manifests / "stage18c_none_followup_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage": "18c_none_followup", "manifest": str(path), "n_jobs": len(rows)}


def build_stage18d_manifest(
    settings: dict,
    outdir: str,
    locked_selection_path: Optional[str] = None,
    allow_placeholder: bool = False,
) -> Dict[str, Any]:
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    lock_path = Path(locked_selection_path or Path(outdir) / "reports" / "round18_locked_selection.json")
    if not lock_path.is_file():
        if not allow_placeholder:
            raise FileNotFoundError(
                f"Missing {lock_path}; Stage 18D requires locked selection from analyzer. "
                "Pass --allow-placeholder-for-smoke only for dry smoke manifests."
            )
        candidates = [
            {"architecture_family": "pooled_mlp", "omics_mode": "own_plus_summary", "architecture_id": "pooled_mlp__own_plus_summary"},
            {"architecture_family": "pooled_transformer", "omics_mode": "own_plus_summary", "architecture_id": "pooled_transformer__P2_standard128__own_plus_summary", "transformer_config_id": "P2_standard128"},
            {"architecture_family": "cross_attention", "omics_mode": "own_plus_summary", "architecture_id": "cross_attn__X1__pure__own_plus_summary", "transformer_config_id": "X1", "residual_mode": "pure"},
            {"architecture_family": "cross_attention", "omics_mode": "own_plus_summary", "architecture_id": "cross_attn__X1__pooled_residual__own_plus_summary", "transformer_config_id": "X1", "residual_mode": "pooled_residual"},
        ]
        lock_source = "placeholder_until_analyzer"
    else:
        lock = load_json(str(lock_path))
        candidates = lock.get("formal_candidates", [])
        if not candidates:
            raise ValueError(f"Lock file has empty formal_candidates: {lock_path}")
        lock_source = str(lock_path)

    n_folds = int(settings["formal_cv"]["n_splits"])
    rows = []
    for cand in candidates:
        for fold_id in range(n_folds):
            rows.append(
                _base_job(
                    job_id=f"18d_{cand['architecture_id']}_f{fold_id}",
                    stage="18d",
                    architecture_id=cand["architecture_id"],
                    architecture_family=cand["architecture_family"],
                    omics_mode=cand["omics_mode"],
                    settings=settings,
                    outdir=outdir,
                    fold_id=fold_id,
                    cv_type="formal_5fold",
                    transformer_config_id=cand.get("transformer_config_id", ""),
                    residual_mode=cand.get("residual_mode", ""),
                )
            )
    path = manifests / "stage18d_formal_5cv_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage": "18d", "manifest": str(path), "n_jobs": len(rows), "lock_source": lock_source}


def build_stage18e_manifest(settings: dict, outdir: str) -> Dict[str, Any]:
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    lock_path = Path(outdir) / "reports" / "round18_locked_selection.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"Missing {lock_path}; Stage 18E requires locked selection after 18D"
        )
    lock = load_json(str(lock_path))
    arch = lock["architecture_id"]
    rows = []
    for fold_id in range(int(settings["formal_cv"]["n_splits"])):
        for target in settings.get("tcga", {}).get("eval_targets", []):
            rows.append(
                {
                    "job_id": f"18e_{arch}_{target['key']}_f{fold_id}",
                    "stage": "18e",
                    "architecture_id": arch,
                    "omics_mode": lock.get("omics_mode", ""),
                    "fold_id": fold_id,
                    "target_key": target["key"],
                    "target_path": target["path"],
                    "mode": "infer_tcga",
                    "result_dir": str(Path(outdir) / "stage18e" / arch / target["key"] / f"fold_{fold_id}"),
                    "status": "pending",
                }
            )
    path = manifests / "stage18e_tcga_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage": "18e", "manifest": str(path), "n_jobs": len(rows)}


def build_stage18f_manifest(settings: dict, outdir: str) -> Dict[str, Any]:
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    lock_path = Path(outdir) / "reports" / "round18_locked_selection.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing {lock_path}; Stage 18F requires locked selection")
    lock = load_json(str(lock_path))
    rows = [
        {
            "job_id": f"18f_export_attention_{lock['architecture_id']}",
            "stage": "18f",
            "architecture_id": lock["architecture_id"],
            "mode": "export_attention",
            "result_dir": str(Path(outdir) / "stage18f" / lock["architecture_id"]),
            "status": "pending",
        }
    ]
    path = manifests / "stage18f_interpretability_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage": "18f", "manifest": str(path), "n_jobs": len(rows)}


def build_round18_configs(
    settings_path: str,
    outdir: str,
    stage: str,
    *,
    allow_placeholder_for_smoke: bool = False,
) -> Dict[str, Any]:
    settings = load_json(settings_path)
    screening_path = Path("config/params_round18_screening.json")
    screening = load_json(str(screening_path)) if screening_path.exists() else {}

    stage_n = stage.lower().replace("stage", "")
    if stage_n in {"18a", "a"}:
        return build_stage18a(settings, outdir)
    if not (Path(outdir) / "splits" / "split_metadata.json").exists():
        build_stage18a(settings, outdir)
    if stage_n in {"18b", "b"}:
        from tools.round18_feature_coverage import assert_round18_feature_coverage

        coverage = assert_round18_feature_coverage(settings)
        cov_path = Path(outdir) / "data" / "round18_feature_coverage_preflight.json"
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        cov_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        out = build_stage18b_manifest(settings, screening, outdir)
        out["feature_coverage"] = str(cov_path)
        return out
    if stage_n in {"18c", "c"}:
        return build_stage18c_manifest(settings, screening, outdir)
    if stage_n in {"18d", "d"}:
        return build_stage18d_manifest(
            settings, outdir, allow_placeholder=allow_placeholder_for_smoke
        )
    if stage_n in {"18e", "e"}:
        return build_stage18e_manifest(settings, outdir)
    if stage_n in {"18f", "f"}:
        return build_stage18f_manifest(settings, outdir)
    raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/round18_architecture_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    parser.add_argument("--stage", required=True, choices=["18a", "18b", "18c", "18d", "18e", "18f", "a", "b", "c", "d", "e", "f"])
    parser.add_argument(
        "--allow-placeholder-for-smoke",
        action="store_true",
        help="Allow Stage 18D placeholder candidates when lock file is missing (smoke only).",
    )
    args = parser.parse_args()
    out = build_round18_configs(
        args.settings,
        args.outdir,
        args.stage,
        allow_placeholder_for_smoke=args.allow_placeholder_for_smoke,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
