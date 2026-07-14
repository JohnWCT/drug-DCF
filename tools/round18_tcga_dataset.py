"""Round 18 TCGA response dataset with Patient_id ↔ latent alignment."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from tools.finetune_tcga_eval import load_tcga_response_csv
from tools.round18_eligible_data import load_smiles_lookup, try_build_graph
from tools.round18_dataset import round18_graph_collate_fn  # re-export compatible collate


def load_tcga_omics_latent_dict(feature_dir: str) -> Dict[str, np.ndarray]:
    path = Path(feature_dir) / "tcga_latent_proto.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TCGA omics latent: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def build_patient_to_latent_key(tcga_latent: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Map short Patient_id (TCGA-XX-XXXX) to first matching latent key."""
    mapping: Dict[str, str] = {}
    for latent_key in tcga_latent.keys():
        parts = str(latent_key).split("-")
        if len(parts) >= 3:
            patient_id = "-".join(parts[:3])
            if patient_id not in mapping:
                mapping[patient_id] = str(latent_key)
    return mapping


def make_eval_row_id(
    *,
    target_key: str,
    patient_id: str,
    drug_name: str,
    label: int,
    source_row_id: int,
) -> str:
    return (
        f"{target_key}|{patient_id}|{str(drug_name).strip()}|{int(label)}|{int(source_row_id)}"
    )


def prepare_tcga_response_frame(
    response_path: str,
    *,
    feature_dir: str,
    drug_smiles_path: str,
    target_key: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, str]]:
    """
    Load TCGA CSV, keep rows with aligned latent + resolvable SMILES.
    Adds columns: Patient_id, DRUG_NAME, Label, eval_row_id, latent_key, _row_id
    """
    raw = load_tcga_response_csv(response_path).copy()
    if "Patient_id" not in raw.columns:
        raise KeyError(f"TCGA CSV missing Patient_id: {response_path}")
    drug_col = "drug_name" if "drug_name" in raw.columns else "DRUG_NAME"
    if drug_col not in raw.columns:
        raise KeyError(f"TCGA CSV missing drug_name: {response_path}")
    if "Label" not in raw.columns:
        raise KeyError(f"TCGA CSV missing Label: {response_path}")

    raw = raw.reset_index(drop=True)
    raw["_source_row_id"] = np.arange(len(raw), dtype=int)
    raw["Patient_id"] = raw["Patient_id"].astype(str)
    raw["DRUG_NAME"] = raw[drug_col].astype(str).str.strip()
    raw["Label"] = raw["Label"].astype(int)

    tcga_latent = load_tcga_omics_latent_dict(feature_dir)
    patient_map = build_patient_to_latent_key(tcga_latent)
    smiles_lookup = load_smiles_lookup(drug_smiles_path)

    rows = []
    n_miss_latent = 0
    n_miss_smiles = 0
    for _, row in raw.iterrows():
        pid = str(row["Patient_id"])
        if pid not in patient_map:
            n_miss_latent += 1
            continue
        drug = str(row["DRUG_NAME"]).strip()
        drug_key = drug.lower()
        smiles = None
        if "smiles" in row.index and pd.notna(row["smiles"]) and str(row["smiles"]).strip():
            smiles = str(row["smiles"]).strip()
        elif drug_key in smiles_lookup:
            smiles = smiles_lookup[drug_key]
        else:
            n_miss_smiles += 1
            continue
        source_row_id = int(row["_source_row_id"])
        label = int(row["Label"])
        rows.append(
            {
                "_source_row_id": source_row_id,
                "Patient_id": pid,
                "ModelID": pid,  # collate / metrics alias
                "DRUG_NAME": drug,
                "drug_name": drug,
                "Label": label,
                "latent_key": patient_map[pid],
                "smiles": smiles,
                "eval_row_id": make_eval_row_id(
                    target_key=target_key,
                    patient_id=pid,
                    drug_name=drug,
                    label=label,
                    source_row_id=source_row_id,
                ),
                "target_key": target_key,
            }
        )
    if not rows:
        raise RuntimeError(
            f"No aligned TCGA rows for {target_key} "
            f"(miss_latent={n_miss_latent}, miss_smiles={n_miss_smiles})"
        )
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=int)
    df.attrs["n_miss_latent"] = n_miss_latent
    df.attrs["n_miss_smiles"] = n_miss_smiles
    return df, tcga_latent, patient_map


class Round18TCGADataset(Dataset):
    """Graph + TCGA omics dataset; ModelID field carries Patient_id for collate reuse."""

    def __init__(
        self,
        response_df: pd.DataFrame,
        *,
        tcga_latent: Dict[str, np.ndarray],
        graph_cache: Optional[Dict[str, Data]] = None,
    ):
        self.df = response_df.reset_index(drop=True).copy()
        if "eval_row_id" not in self.df.columns:
            raise KeyError("Round18TCGADataset requires eval_row_id")
        if "latent_key" not in self.df.columns:
            raise KeyError("Round18TCGADataset requires latent_key")
        self.tcga_latent = tcga_latent
        sample = next(iter(tcga_latent.values()))
        self.omics_dim = int(np.asarray(sample).reshape(-1).shape[0])
        self.graph_cache = graph_cache if graph_cache is not None else {}
        self._ensure_graphs()
        self.df["sample_weight"] = 1.0

    def _ensure_graphs(self) -> None:
        for _, row in self.df.iterrows():
            key = str(row["DRUG_NAME"]).strip().lower()
            if key in self.graph_cache:
                continue
            smiles = str(row["smiles"])
            ok, graph, info = try_build_graph(smiles)
            if not ok or graph is None:
                raise RuntimeError(f"Graph build failed for {key}: {info.get('error')}")
            graph.original_smiles = info.get("original_smiles")
            graph.desalt_applied = bool(info.get("desalt_applied"))
            graph.fragment_count = int(info.get("fragment_count") or 0)
            self.graph_cache[key] = graph

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        latent_key = str(row["latent_key"])
        if latent_key not in self.tcga_latent:
            raise KeyError(f"Missing TCGA latent key={latent_key}")
        omics = torch.tensor(np.asarray(self.tcga_latent[latent_key], dtype=np.float32).reshape(-1))
        assert omics.numel() == self.omics_dim
        drug_key = str(row["DRUG_NAME"]).strip().lower()
        return {
            "_row_id": int(row["_row_id"]),
            "ModelID": str(row["Patient_id"]),
            "DRUG_NAME": str(row["DRUG_NAME"]),
            "Label": int(row["Label"]),
            "weight": float(row["sample_weight"]),
            "omics": omics,
            "drug_graph": self.graph_cache[drug_key],
            "eval_row_id": str(row["eval_row_id"]),
            "Patient_id": str(row["Patient_id"]),
            "target_key": str(row.get("target_key", "")),
        }


def round18_tcga_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = round18_graph_collate_fn(batch)
    out["eval_row_id"] = [b["eval_row_id"] for b in batch]
    out["Patient_id"] = [b["Patient_id"] for b in batch]
    out["target_key"] = [b.get("target_key", "") for b in batch]
    return out
