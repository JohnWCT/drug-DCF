"""BioCDA XA validation dataset — splits 96-d O2 latent into Z64 + C32."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from tools.round19_dataset import Round19ResponseDataset, round19_collate_fn


def split_z_context(latent96: torch.Tensor, *, z_dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    if latent96.shape[-1] < z_dim + 1:
        raise ValueError(f"Expected latent dim >= {z_dim + 1}, got {latent96.shape[-1]}")
    return latent96[..., :z_dim], latent96[..., z_dim:]


def biocda_collate_fn(batch: list) -> dict:
    base = round19_collate_fn(batch)
    omics96 = base["omics"]
    omics, context = split_z_context(omics96)
    drug_graph = base.get("drug_batch")
    if drug_graph is None:
        raise ValueError("BioCDA collate requires graph drug_batch")
    return {
        "omics": omics,
        "context": context,
        "labels": base["label"],
        "weights": base.get("weight"),
        "drug_graph": drug_graph,
        "_row_id": base["_row_id"],
        "ModelID": base["ModelID"],
        "DRUG_NAME": base["DRUG_NAME"],
    }


def build_xa_dataset(
    response_df: pd.DataFrame,
    *,
    feature_dir: str,
    drug_smiles_path: str,
    graph_cache: Optional[dict] = None,
) -> Round19ResponseDataset:
    return Round19ResponseDataset(
        response_df,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
        encoder_type="gin",
        graph_cache=graph_cache if graph_cache is not None else {},
    )


def row_ids_for_role(assignments: pd.DataFrame, *, split_seed: int, role: str) -> list[int]:
    part = assignments[
        (assignments["split_seed"] == int(split_seed)) & (assignments["split_role"] == role)
    ]
    return part["_row_id"].astype(int).tolist()


def build_loaders(
    dataset: Round19ResponseDataset,
    assignments: pd.DataFrame,
    *,
    split_seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ids = row_ids_for_role(assignments, split_seed=split_seed, role="train")
    val_ids = row_ids_for_role(assignments, split_seed=split_seed, role="val")
    test_ids = row_ids_for_role(assignments, split_seed=split_seed, role="test")
    id_to_idx = {int(rid): i for i, rid in enumerate(dataset.df["_row_id"].astype(int))}

    def _subset(row_ids: list[int]) -> Subset:
        indices = [id_to_idx[r] for r in row_ids if r in id_to_idx]
        return Subset(dataset, indices)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": biocda_collate_fn,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return (
        DataLoader(_subset(train_ids), shuffle=True, **kwargs),
        DataLoader(_subset(val_ids), shuffle=False, **kwargs),
        DataLoader(_subset(test_ids), shuffle=False, **kwargs),
    )
