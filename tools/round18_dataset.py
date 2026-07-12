"""Round 18 Dataset / collate / feature loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from tools.round18_eligible_data import (
    load_omics_latent_dict,
    load_smiles_lookup,
    try_build_graph,
    validate_feature_metadata,
)


class Round18ResponseDataset(Dataset):
    def __init__(
        self,
        response_df: pd.DataFrame,
        *,
        feature_dir: str,
        drug_smiles_path: str,
        group_column: str = "ModelID",
        label_column: str = "Label",
        drug_column: str = "DRUG_NAME",
        graph_cache: Optional[Dict[str, Data]] = None,
    ):
        self.df = response_df.reset_index(drop=True).copy()
        if "_row_id" not in self.df.columns:
            self.df["_row_id"] = np.arange(len(self.df), dtype=int)
        self.group_column = group_column
        self.label_column = label_column
        self.drug_column = drug_column if drug_column in self.df.columns else "DRUG_NAME"

        self.feature_dir = feature_dir
        self.feature_meta = validate_feature_metadata(feature_dir)
        self.latent = load_omics_latent_dict(feature_dir)
        sample = next(iter(self.latent.values()))
        self.omics_dim = int(np.asarray(sample).reshape(-1).shape[0])
        if self.feature_meta.get("response_input_dim") is not None:
            assert int(self.feature_meta["response_input_dim"]) == self.omics_dim

        self.smiles_lookup = load_smiles_lookup(drug_smiles_path)
        self.graph_cache = graph_cache if graph_cache is not None else {}
        self._ensure_graphs()
        self._compute_sample_weights()

    def _drug_key(self, row) -> str:
        if "drug_smiles_key" in row.index and pd.notna(row["drug_smiles_key"]):
            return str(row["drug_smiles_key"]).lower()
        return str(row[self.drug_column]).strip().lower()

    def _ensure_graphs(self) -> None:
        for _, row in self.df.iterrows():
            key = self._drug_key(row)
            if key in self.graph_cache:
                continue
            if key not in self.smiles_lookup:
                raise KeyError(f"Missing SMILES for drug key={key}")
            ok, graph, info = try_build_graph(self.smiles_lookup[key])
            if not ok or graph is None:
                raise RuntimeError(f"Graph build failed for {key}: {info.get('error')}")
            # attach interpretability fields on Data object
            graph.original_smiles = info.get("original_smiles")
            graph.desalt_applied = bool(info.get("desalt_applied"))
            graph.fragment_count = int(info.get("fragment_count") or 0)
            self.graph_cache[key] = graph

    def _compute_sample_weights(self) -> None:
        from sklearn.utils.class_weight import compute_class_weight

        group = (
            self.df[self.drug_column].astype(str)
            + "_"
            + self.df[self.label_column].astype(str)
        )
        classes = np.unique(group)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=group)
        wmap = dict(zip(classes, weights))
        self.df["sample_weight"] = group.map(wmap).astype("float32")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        mid = str(row[self.group_column])
        if mid not in self.latent:
            raise KeyError(f"Missing latent for ModelID={mid}")
        omics = torch.tensor(np.asarray(self.latent[mid], dtype=np.float32).reshape(-1))
        assert omics.numel() == self.omics_dim
        drug_key = self._drug_key(row)
        graph = self.graph_cache[drug_key]
        return {
            "_row_id": int(row["_row_id"]),
            "ModelID": mid,
            "DRUG_NAME": str(row[self.drug_column]),
            "Label": int(row[self.label_column]),
            "weight": float(row["sample_weight"]),
            "omics": omics,
            "drug_graph": graph,
        }


def round18_graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    omics = torch.stack([b["omics"] for b in batch], dim=0)
    labels = torch.tensor([b["Label"] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    drug_batch = Batch.from_data_list([b["drug_graph"] for b in batch])
    return {
        "_row_id": [b["_row_id"] for b in batch],
        "ModelID": [b["ModelID"] for b in batch],
        "DRUG_NAME": [b["DRUG_NAME"] for b in batch],
        "drug_name": [b["DRUG_NAME"] for b in batch],
        "label": labels,
        "weight": weights,
        "omics": omics,
        "drug_batch": drug_batch,
    }


def build_round18_drug_graph_cache(drug_smiles_path: str, drug_keys: Sequence[str]) -> Dict[str, Data]:
    lookup = load_smiles_lookup(drug_smiles_path)
    cache: Dict[str, Data] = {}
    for key in drug_keys:
        k = str(key).strip().lower()
        if k not in lookup:
            continue
        ok, graph, info = try_build_graph(lookup[k])
        if ok and graph is not None:
            graph.original_smiles = info.get("original_smiles")
            graph.desalt_applied = bool(info.get("desalt_applied"))
            graph.fragment_count = int(info.get("fragment_count") or 0)
            cache[k] = graph
    return cache


def validate_round18_row_alignment(df: pd.DataFrame, latent_keys, smiles_keys) -> None:
    missing_latent = set(df["ModelID"].astype(str)) - set(map(str, latent_keys))
    if missing_latent:
        raise AssertionError(f"rows missing latent: {sorted(list(missing_latent))[:5]}")
    drug_col = "DRUG_NAME" if "DRUG_NAME" in df.columns else "mapped_name"
    missing_smiles = set(df[drug_col].astype(str).str.lower()) - set(map(str, smiles_keys))
    if missing_smiles:
        raise AssertionError(f"rows missing smiles: {sorted(list(missing_smiles))[:5]}")


def subset_by_assignment(
    eligible_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    *,
    fold_id: int,
    split_role: str,
) -> pd.DataFrame:
    rows = assignment_df[
        (assignment_df["fold_id"] == int(fold_id)) & (assignment_df["split_role"] == split_role)
    ]["_row_id"].astype(int)
    out = eligible_df[eligible_df["_row_id"].isin(set(rows))].copy()
    if len(out) == 0:
        raise ValueError(f"Empty subset fold={fold_id} role={split_role}")
    return out
