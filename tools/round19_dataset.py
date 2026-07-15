"""Round 19 dataset: graph (GIN/GINE) or MACCS branches."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from tools.round18_dataset import Round18ResponseDataset, subset_by_assignment
from tools.round18_eligible_data import load_omics_latent_dict, load_smiles_lookup, validate_feature_metadata
from tools.round19_drug_features import load_maccs_by_drug_name, validate_maccs_coverage
from tools.round19_graph_features import (
    build_pyg_data,
    graph_smiles_identity,
    legacy_graph_metadata,
)


def _same_legacy_molecule(left: str, right: str) -> bool:
    """Allow notation differences only when RDKit canonical identities agree."""
    return (
        legacy_graph_metadata(left)["graph_smiles_canonical_identity"]
        == legacy_graph_metadata(right)["graph_smiles_canonical_identity"]
    )


class Round19ResponseDataset(Dataset):
    """Omics + drug representation dataset for Round 19 smoke/training."""

    def __init__(
        self,
        response_df: pd.DataFrame,
        *,
        feature_dir: str,
        drug_smiles_path: str,
        encoder_type: str = "gin",
        with_bonds: bool = False,
        maccs_by_drug: Optional[Dict[str, np.ndarray]] = None,
        graph_cache: Optional[Dict[str, Data]] = None,
        context_permutation: Optional[Dict[str, str]] = None,
        omics_id: Optional[str] = None,
        latent_by_id: Optional[Dict[str, np.ndarray]] = None,
        group_column: str = "ModelID",
        label_column: str = "Label",
        drug_column: str = "DRUG_NAME",
    ):
        self.encoder_type = str(encoder_type).lower()
        self.with_bonds = bool(with_bonds)
        if self.encoder_type == "maccs" and self.with_bonds:
            raise AssertionError("MACCS cannot request bond graphs")
        if self.encoder_type in {"gin", "gine"} and maccs_by_drug is not None:
            raise AssertionError("Hybrid forbidden: graph encoder with MACCS map")

        self.df = response_df.reset_index(drop=True).copy()
        if "_row_id" not in self.df.columns:
            self.df["_row_id"] = np.arange(len(self.df), dtype=int)
        self.group_column = group_column
        self.label_column = label_column
        self.drug_column = drug_column if drug_column in self.df.columns else "DRUG_NAME"

        self.feature_dir = feature_dir
        if latent_by_id is None:
            self.feature_meta = validate_feature_metadata(feature_dir)
            self.latent = load_omics_latent_dict(feature_dir)
        else:
            self.feature_meta = {"external_latent_override": True}
            self.latent = {
                str(key): np.asarray(value, dtype=np.float32).reshape(-1)
                for key, value in latent_by_id.items()
            }
            if not self.latent:
                raise ValueError("latent_by_id must not be empty")
        sample = next(iter(self.latent.values()))
        self.omics_dim = int(np.asarray(sample).reshape(-1).shape[0])

        self.smiles_lookup = load_smiles_lookup(drug_smiles_path)
        self.graph_cache = graph_cache if graph_cache is not None else {}
        self.maccs_by_drug = maccs_by_drug
        self.context_permutation = context_permutation
        self.omics_id = str(omics_id) if omics_id is not None else None

        if self.encoder_type == "maccs":
            drugs = sorted(set(self.df[self.drug_column].astype(str)))
            if self.maccs_by_drug is None:
                self.maccs_by_drug = load_maccs_by_drug_name(drug_smiles_path, drug_names=drugs)
            validate_maccs_coverage(self.maccs_by_drug, drugs)
        else:
            self._ensure_graphs()

        tmp = Round18ResponseDataset.__new__(Round18ResponseDataset)
        tmp.df = self.df
        tmp.drug_column = self.drug_column
        tmp.label_column = self.label_column
        tmp._compute_sample_weights()
        self.df = tmp.df

    def _drug_key(self, row) -> str:
        if "drug_smiles_key" in row.index and pd.notna(row["drug_smiles_key"]):
            return str(row["drug_smiles_key"]).lower()
        return str(row[self.drug_column]).strip().lower()

    def _ensure_graphs(self) -> None:
        for _, row in self.df.iterrows():
            key = self._drug_key(row)
            inline_smiles = (
                str(row["smiles"]).strip()
                if "smiles" in row.index and pd.notna(row["smiles"])
                else ""
            )
            if key not in self.smiles_lookup and not inline_smiles:
                raise KeyError(f"Missing SMILES for drug key={key}")
            lookup_smiles = str(self.smiles_lookup.get(key, "")).strip()
            if (
                inline_smiles
                and lookup_smiles
                and inline_smiles != lookup_smiles
                and not _same_legacy_molecule(inline_smiles, lookup_smiles)
            ):
                raise AssertionError(
                    f"Conflicting inline/lookup SMILES for drug key={key}; refusing ambiguous graph source"
                )
            smiles = lookup_smiles or inline_smiles
            source = "lookup" if lookup_smiles else "inline"
            cache_key = (
                f"{key}|graph={graph_smiles_identity(smiles)}|"
                f"encoder={self.encoder_type}|bonds={int(self.with_bonds)}"
            )
            if cache_key in self.graph_cache:
                continue
            graph = build_pyg_data(smiles, with_bonds=self.with_bonds)
            graph.actual_smiles_source = source
            graph.graph_cache_key = cache_key
            self.graph_cache[cache_key] = graph

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        mid = str(row[self.group_column])
        if mid not in self.latent:
            raise KeyError(f"Missing omics for ModelID={mid}")
        omics_arr = np.asarray(self.latent[mid], dtype=np.float32)
        if self.context_permutation is not None:
            from tools.round19_context_controls import apply_context_permutation

            donor_mid = self.context_permutation.get(mid, mid)
            if donor_mid not in self.latent:
                raise KeyError(f"Missing donor omics for ModelID={donor_mid}")
            donor_arr = np.asarray(self.latent[donor_mid], dtype=np.float32)
            oid = self.omics_id or self.feature_meta.get("omics_id", "O2")
            omics_arr = apply_context_permutation(omics_arr, donor_arr, oid)
        omics = torch.tensor(omics_arr)
        drug_name = str(row[self.drug_column])
        item = {
            "_row_id": int(row["_row_id"]),
            "ModelID": mid,
            "DRUG_NAME": drug_name,
            "drug_name": drug_name,
            "Label": int(row[self.label_column]),
            "weight": float(self.df.iloc[idx]["sample_weight"]),
            "omics": omics,
        }
        for column in ("eval_row_id", "Patient_id", "target_key"):
            if column in row.index:
                item[column] = str(row[column])
        if self.encoder_type == "maccs":
            item["maccs"] = torch.tensor(self.maccs_by_drug[drug_name], dtype=torch.float32)
            item["drug_graph"] = None
        else:
            key = self._drug_key(row)
            inline_smiles = (
                str(row["smiles"]).strip()
                if "smiles" in row.index and pd.notna(row["smiles"])
                else ""
            )
            lookup_smiles = str(self.smiles_lookup.get(key, "")).strip()
            if (
                inline_smiles
                and lookup_smiles
                and inline_smiles != lookup_smiles
                and not _same_legacy_molecule(inline_smiles, lookup_smiles)
            ):
                raise AssertionError(f"Conflicting inline/lookup SMILES for drug key={key}")
            smiles = lookup_smiles or inline_smiles
            cache_key = (
                f"{key}|graph={graph_smiles_identity(smiles)}|"
                f"encoder={self.encoder_type}|bonds={int(self.with_bonds)}"
            )
            graph = self.graph_cache[cache_key]
            item["drug_graph"] = graph
            item["graph_smiles"] = graph.graph_smiles
            item["legacy_input_smiles"] = graph.legacy_input_smiles
            item["actual_smiles_source"] = graph.actual_smiles_source
            item["graph_metadata"] = graph.graph_metadata
            item["maccs"] = None
        return item


def round19_collate_fn(batch: list) -> dict:
    omics = torch.stack([b["omics"] for b in batch], dim=0)
    labels = torch.tensor([b["Label"] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    out = {
        "omics": omics,
        "label": labels,
        "weight": weights,
        "_row_id": [b["_row_id"] for b in batch],
        "ModelID": [b["ModelID"] for b in batch],
        "DRUG_NAME": [b["DRUG_NAME"] for b in batch],
        "drug_name": [b["drug_name"] for b in batch],
    }
    for column in ("eval_row_id", "Patient_id", "target_key"):
        if all(column in b for b in batch):
            out[column] = [b[column] for b in batch]
    if batch[0]["maccs"] is None:
        for column in (
            "graph_smiles",
            "legacy_input_smiles",
            "actual_smiles_source",
            "graph_metadata",
        ):
            out[column] = [b[column] for b in batch]
    if batch[0]["maccs"] is not None:
        out["maccs"] = torch.stack([b["maccs"] for b in batch], dim=0)
        out["drug_batch"] = None
    else:
        out["maccs"] = None
        out["drug_batch"] = Batch.from_data_list([b["drug_graph"] for b in batch])
    return out


__all__ = [
    "Round19ResponseDataset",
    "round19_collate_fn",
    "subset_by_assignment",
]
