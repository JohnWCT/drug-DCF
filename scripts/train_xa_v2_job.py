#!/usr/bin/env python3
"""Single Round-23 XA v2 training job (fresh / transfer / KD / z-only / predictive eval)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch.multiprocessing.set_sharing_strategy("file_system")

from biocda.data.xa_dataset import build_loaders, build_xa_dataset, row_ids_for_role
from biocda.models.predictive import load_biocda_predictive
from biocda.models.xa.factory import build_xa_v2
from biocda.training.freeze_schedule import FreezePhase
from biocda.training.gin_transfer import transfer_e3_gin_to_xa, write_transfer_report
from biocda.training.graph_cache_io import ensure_graph_cache, load_graph_cache
from biocda.training.xa_loop import _predict, metrics_to_summary_row, pos_weight_from_labels
from biocda.training.xa_v2_trainer import train_xa_v2_run
from biocda.utils.gpu import build_efficient_dataloader_kwargs, configure_gpu_efficiency
from biocda.utils.hashing import sha256_json
from biocda.utils.reproducibility import set_seed
from biocda.utils.runtime_manifest import build_run_manifest, write_run_manifest
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configure_gpu_efficiency(target_utilization=float(config["training"].get("target_gpu_utilization", 0.9)))

    out_root = ROOT / config["outputs"]["root"]
    ensure_graph_cache(
        dev_rows_path=ROOT / config["data"]["development_rows"],
        feature_dir=str(ROOT / config["data"]["feature_dir"]),
        drug_smiles_path=str(ROOT / config["data"]["drug_smiles_path"]),
        cache_root=out_root,
    )
    graph_cache = load_graph_cache(out_root)
    assignments = pd.read_csv(ROOT / config["data"]["assignments_csv"])
    dev = pd.read_csv(ROOT / config["data"]["development_rows"])
    dataset = build_xa_dataset(
        dev,
        feature_dir=str(ROOT / config["data"]["feature_dir"]),
        drug_smiles_path=str(ROOT / config["data"]["drug_smiles_path"]),
        graph_cache=graph_cache,
    )

    dl_kwargs = build_efficient_dataloader_kwargs(
        batch_size=int(config["training"].get("micro_batch_size", 512)),
    )
    nw = int(config["training"].get("dataloader_num_workers", 0))
    dl_kwargs["num_workers"] = nw
    if nw == 0:
        dl_kwargs.pop("persistent_workers", None)
        dl_kwargs.pop("prefetch_factor", None)

    set_seed(args.seed)
    run_dir = out_root / f"{args.model_type}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(
        dataset,
        assignments,
        split_seed=args.seed,
        batch_size=dl_kwargs["batch_size"],
        num_workers=dl_kwargs["num_workers"],
        pin_memory=dl_kwargs["pin_memory"],
    )
    train_ids = row_ids_for_role(assignments, split_seed=args.seed, role="train")
    id_to_idx = {int(rid): i for i, rid in enumerate(dataset.df["_row_id"].astype(int))}
    train_labels = torch.tensor(
        [int(dataset.df.iloc[id_to_idx[r]]["Label"]) for r in train_ids if r in id_to_idx],
        dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pw = pos_weight_from_labels(train_labels, device)

    pred_ckpt = ROOT / config["data"]["predictive_checkpoint"]
    smoke = args.smoke
    tr = config["training"]
    if smoke:
        phases = [
            FreezePhase("attention_warmup", epochs=int(tr.get("smoke_epochs", 2)), freeze_gin_layers=[0, 1, 2, 3, 4], other_lr=3e-4),
        ]
        patience = 2
    else:
        phases = [
            FreezePhase("attention_warmup", epochs=int(tr.get("warmup_epochs", 15)), freeze_gin_layers=[0, 1, 2, 3, 4], other_lr=3e-4),
            FreezePhase(
                "last_gin_adaptation",
                epochs=int(tr.get("last_gin_epochs", 40)),
                freeze_gin_layers=[0, 1, 2, 3],
                last_gin_lr=float(config["optimizer"].get("last_gin_learning_rate", 1e-5)),
                other_lr=float(config["optimizer"].get("learning_rate", 3e-4)),
            ),
            FreezePhase(
                "joint_stabilization",
                epochs=int(tr.get("stabilize_epochs", 145)),
                freeze_gin_layers=[0, 1, 2, 3],
                last_gin_lr=float(config["optimizer"].get("last_gin_learning_rate", 1e-5)),
                other_lr=float(config["optimizer"].get("learning_rate", 3e-4)),
            ),
        ]
        patience = int(tr["early_stopping_patience"])

    # --- P0: train BioCDA-Predictive (pooled E3) on the SAME split ---
    # Round20 checkpoint is LOCKED for transfer/KD teacher only; evaluating that
    # checkpoint on Round21 seeds would be unpaired / leakage-prone.
    if args.model_type == "biocda_predictive":
        from biocda.models.predictive.pooled_e3 import BioCDAPredictive
        from biocda.training.xa_loop import train_xa_run

        model = BioCDAPredictive()
        # Optional: warm-start GIN from Round20 (still train fusion/head on this split)
        if bool(config.get("training", {}).get("predictive_warmstart_gin", True)):
            from biocda.training.gin_transfer import transfer_e3_gin_into_module

            transfer_e3_gin_into_module(pred_ckpt, model.encoder, strict=True)

        max_epochs = int(tr.get("smoke_epochs", 2)) if smoke else int(tr["max_epochs"])
        result = train_xa_run(
            model,
            train_loader,
            val_loader,
            test_loader,
            run_dir=run_dir,
            max_epochs=max_epochs,
            patience=2 if smoke else int(tr["early_stopping_patience"]),
            lr=float(config["optimizer"].get("learning_rate", 3e-4)),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            grad_clip=float(tr["gradient_clip_norm"]),
            use_amp=bool(tr.get("mixed_precision", True)),
            accumulation_steps=int(tr.get("accumulation_steps", 1)),
            model_type="biocda_predictive",
            architecture_version="biocda-predictive-e3",
            config=config,
            pos_weight=pw,
        )
        print(f"JOB_OK model={args.model_type} seed={args.seed} auc={result.metrics_validation.get('DrugMacro_AUC')}")
        return

    xa_type = args.model_type
    model = build_xa_v2(config, model_type=xa_type if xa_type != "biocda_xa_kd" else "biocda_xa_kd")
    # KD and transfer share architecture; z_only / fresh as named
    if xa_type in ("biocda_xa_transfer", "biocda_xa_kd"):
        report = transfer_e3_gin_to_xa(pred_ckpt, model, strict=True)
        write_transfer_report(run_dir / "gin_transfer_report.json", report)

    teacher = None
    if xa_type == "biocda_xa_kd":
        teacher = load_biocda_predictive(pred_ckpt)

    kd_cfg = tr.get("distillation", {})
    result = train_xa_v2_run(
        model,
        train_loader,
        val_loader,
        test_loader,
        run_dir=run_dir,
        phases=phases,
        patience=patience,
        weight_decay=float(config["optimizer"]["weight_decay"]),
        grad_clip=float(tr["gradient_clip_norm"]),
        use_amp=bool(tr.get("mixed_precision", True)),
        accumulation_steps=int(tr.get("accumulation_steps", 1)),
        model_type=xa_type,
        architecture_version=config["experiment"]["architecture_version"],
        config=config,
        pos_weight=pw,
        teacher=teacher,
        lambda_kd=float(kd_cfg.get("lambda_kd", 0.5)),
        kd_temperature=float(kd_cfg.get("temperature", 2.0)),
    )
    write_run_manifest(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command=f"train_xa_v2_job {xa_type} seed={args.seed}",
            config=config,
            config_hash=sha256_json(config),
            seed=args.seed,
        ),
    )
    print(f"JOB_OK model={xa_type} seed={args.seed} auc={result.metrics_validation.get('DrugMacro_AUC')}")


if __name__ == "__main__":
    main()
