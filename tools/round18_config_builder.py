"""Round 18 config / manifest builder."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_splits import write_round18_splits


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_dir_for_omics(settings: dict, omics_mode: str) -> str:
    root = settings.get("feature_root", "result/optimization_runs/round17r_18class/features")
    model_key = settings.get("feature_model_key", "r13_exp_008")
    mode = "none" if omics_mode in {"none", "z-only", "z_only"} else omics_mode
    return str(Path(root) / model_key / mode)


def build_stage18a(settings: dict, outdir: str) -> Dict[str, Any]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = write_round18_splits(
        settings["response_data_path"],
        outdir,
        group_column=settings["internal_test"]["group_column"],
        label_column=settings["internal_test"]["label_column"],
        split_seed=int(settings.get("split_seed", 42)),
        screening_folds=int(settings["screening_cv"]["n_splits"]),
        formal_folds=int(settings["formal_cv"]["n_splits"]),
    )
    meta = {
        "stage": "18a",
        "outdir": str(out),
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
    return {"stage": "18a", "metadata": str(meta_path), **paths}


def build_stage18b_manifest(settings: dict, screening: dict, outdir: str) -> Dict[str, Any]:
    """Create screening job rows for MLP + pooled Transformer (no training yet)."""
    manifests = Path(outdir) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    omics_modes = settings.get("omics_modes", ["none", "own_plus_summary", "own_proto_context_projected_16"])
    n_folds = int(settings["screening_cv"]["n_splits"])
    rows: List[dict] = []

    for omics in omics_modes:
        for fold_id in range(n_folds):
            rows.append(
                {
                    "job_id": f"18b_mlp_{omics}_f{fold_id}",
                    "stage": "18b",
                    "architecture_id": f"pooled_mlp__{omics}",
                    "architecture_family": "pooled_mlp",
                    "omics_mode": omics,
                    "transformer_config_id": "",
                    "residual_mode": "",
                    "cv_type": "screening_3fold",
                    "fold_id": fold_id,
                    "model_seed": settings.get("model_seed", 101),
                    "split_seed": settings.get("split_seed", 42),
                    "drug_smiles_path": settings["drug_smiles_path"],
                    "response_data_path": settings["response_data_path"],
                    "feature_dir": feature_dir_for_omics(settings, omics),
                    "result_dir": str(Path(outdir) / "stage18b" / f"pooled_mlp__{omics}" / f"fold_{fold_id}"),
                    "requested_micro_batch": settings["oom"]["micro_batch_candidates"][0],
                    "target_effective_batch": settings["oom"]["target_effective_batch"],
                }
            )

    for cfg in screening.get("pooled_transformer_configs", []):
        cfg_id = cfg["config_id"]
        for omics in omics_modes:
            for fold_id in range(n_folds):
                rows.append(
                    {
                        "job_id": f"18b_tf_{cfg_id}_{omics}_f{fold_id}",
                        "stage": "18b",
                        "architecture_id": f"pooled_transformer__{cfg_id}__{omics}",
                        "architecture_family": "pooled_transformer",
                        "omics_mode": omics,
                        "transformer_config_id": cfg_id,
                        "residual_mode": "",
                        "cv_type": "screening_3fold",
                        "fold_id": fold_id,
                        "model_seed": settings.get("model_seed", 101),
                        "split_seed": settings.get("split_seed", 42),
                        "drug_smiles_path": settings["drug_smiles_path"],
                        "response_data_path": settings["response_data_path"],
                        "feature_dir": feature_dir_for_omics(settings, omics),
                        "result_dir": str(
                            Path(outdir) / "stage18b" / f"pooled_transformer__{cfg_id}__{omics}" / f"fold_{fold_id}"
                        ),
                        "requested_micro_batch": settings["oom"]["micro_batch_candidates"][0],
                        "target_effective_batch": settings["oom"]["target_effective_batch"],
                    }
                )

    import pandas as pd

    path = manifests / "stage18b_screening_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"stage": "18b", "manifest": str(path), "n_jobs": len(rows)}


def build_round18_configs(settings_path: str, outdir: str, stage: str) -> Dict[str, Any]:
    settings = load_json(settings_path)
    screening_path = Path("config/params_round18_screening.json")
    screening = load_json(str(screening_path)) if screening_path.exists() else {}

    stage_n = stage.lower().replace("stage", "")
    if stage_n in {"18a", "a"}:
        return build_stage18a(settings, outdir)
    if stage_n in {"18b", "b"}:
        # ensure splits exist
        if not (Path(outdir) / "splits" / "split_metadata.json").exists():
            build_stage18a(settings, outdir)
        return build_stage18b_manifest(settings, screening, outdir)
    raise ValueError(f"Unsupported stage for builder smoke: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/round18_architecture_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    parser.add_argument("--stage", required=True, choices=["18a", "18b", "18c", "18d", "18e", "18f", "a", "b", "c", "d", "e", "f"])
    args = parser.parse_args()
    out = build_round18_configs(args.settings, args.outdir, args.stage)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
