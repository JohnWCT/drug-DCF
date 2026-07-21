#!/usr/bin/env python3
"""Single BioCDA XA training job worker (for parallel dispatch)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from biocda.data.xa_dataset import build_loaders, build_xa_dataset, row_ids_for_role
from biocda.models.model_factory import build_model, export_freeze_policy
from biocda.training.graph_cache_io import ensure_graph_cache, load_graph_cache
from biocda.training.xa_loop import pos_weight_from_labels, train_xa_run
from biocda.utils.gpu import build_efficient_dataloader_kwargs, configure_gpu_efficiency
from biocda.utils.hashing import sha256_json
from biocda.utils.reproducibility import set_seed
from biocda.utils.runtime_manifest import build_run_manifest, write_run_manifest


def _model_config_for(model_type: str, base_path: Path) -> dict:
    from biocda.models.model_factory import build_model_config_for_type

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg = build_model_config_for_type(base, model_type)
    cfg["training"]["freeze_policy"] = {
        "omics_encoder": True,
        "sample_encoder": False,
        "drug_encoder": True,
        "cross_attention": False,
        "response_head": False,
        "fusion": True,
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
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

    cfg = _model_config_for(args.model_type, ROOT / "configs/model/biocda_cross_attention.yaml")
    set_seed(args.seed)
    model = build_model(cfg)
    run_dir = out_root / f"{args.model_type}_seed{args.seed}"

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

    result = train_xa_run(
        model,
        train_loader,
        val_loader,
        test_loader,
        run_dir=run_dir,
        max_epochs=int(config["training"]["max_epochs"]),
        patience=int(config["training"]["early_stopping_patience"]),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
        grad_clip=float(config["training"]["gradient_clip_norm"]),
        use_amp=bool(config["training"].get("mixed_precision", True)),
        accumulation_steps=int(config["training"].get("accumulation_steps", 1)),
        model_type=args.model_type,
        architecture_version=config["experiment"]["architecture_version"],
        config=cfg,
        pos_weight=pw,
    )
    export_freeze_policy(model, run_dir / "freeze_policy.json")
    write_run_manifest(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command=f"run_xa_train_job {args.model_type} seed={args.seed}",
            config=config,
            config_hash=sha256_json(config),
            seed=args.seed,
        ),
    )
    print(f"JOB_OK model={args.model_type} seed={args.seed} auc={result.metrics_validation.get('DrugMacro_AUC')}")


if __name__ == "__main__":
    main()
