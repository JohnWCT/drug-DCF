"""Build Round 18 eligible response rows before CV splitting."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple  # noqa: F401 — Dict used by graph cache maps

import numpy as np
import pandas as pd
import torch
from torch_geometric import data as DATA

from tools.dataprocess import smile_to_graph


def _normalize_drug_key(name: str) -> str:
    return str(name).strip().lower()


def load_omics_latent_dict(feature_dir: str) -> Dict[str, np.ndarray]:
    path = Path(feature_dir) / "ccle_latent_proto.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing omics latent: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def validate_feature_metadata(feature_dir: str, expected_dim: Optional[int] = None) -> Dict[str, Any]:
    meta_path = Path(feature_dir) / "feature_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature_metadata.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_types = meta.get("n_trainable_cancer_types")
    if n_types is not None and int(n_types) != 18:
        raise AssertionError(f"expected n_trainable_cancer_types=18, got {n_types}")
    if meta.get("uses_legacy_28class_cache") is True:
        raise AssertionError("feature artifact uses_legacy_28class_cache=true")
    src = meta.get("prototype_class_source")
    if src is not None and src != "checkpoint_metadata":
        raise AssertionError(f"unexpected prototype_class_source={src}")
    dim = meta.get("response_input_dim")
    if expected_dim is not None and dim is not None and int(dim) != int(expected_dim):
        raise AssertionError(f"response_input_dim mismatch: meta={dim} expected={expected_dim}")
    return meta


def load_smiles_lookup(drug_smiles_path: str) -> Dict[str, str]:
    df = pd.read_csv(drug_smiles_path)
    # Prefer lowercase drug_name index used by existing finetune utils
    lookup: Dict[str, str] = {}
    name_cols = [c for c in ("drug_name", "DRUG_NAME", "name") if c in df.columns]
    if "SMILES" not in df.columns:
        raise KeyError(f"SMILES column missing in {drug_smiles_path}")
    for _, row in df.iterrows():
        smiles = row["SMILES"]
        if pd.isna(smiles) or not str(smiles).strip():
            continue
        for col in name_cols:
            key = _normalize_drug_key(row[col])
            if key and key not in lookup:
                lookup[key] = str(smiles)
        # also index values
        if df.index.name or not isinstance(df.index, pd.RangeIndex):
            key = _normalize_drug_key(row.name if not isinstance(row.name, (int, np.integer)) else "")
            if key:
                lookup.setdefault(key, str(smiles))
    # index-based lowercase keys
    for idx, smiles in zip(df.index.astype(str), df["SMILES"].astype(str)):
        lookup.setdefault(_normalize_drug_key(idx), smiles)
    return lookup


def try_build_graph(smiles: str) -> Tuple[bool, Optional[DATA.Data], Dict[str, Any]]:
    info = {
        "original_smiles": smiles,
        "desalt_applied": "." in smiles,
        "fragment_count": smiles.count(".") + 1 if smiles else 0,
        "graph_smiles": None,
        "n_atoms": 0,
        "error": None,
    }
    try:
        c_size, feats, edge_index = smile_to_graph(smiles)
        if c_size <= 0 or not feats:
            info["error"] = "empty_graph"
            return False, None, info
        x = torch.tensor(np.asarray(feats), dtype=torch.float32)
        if edge_index:
            ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        data = DATA.Data(x=x, edge_index=ei)
        info["n_atoms"] = int(c_size)
        info["graph_smiles"] = smiles  # smile_to_graph keeps largest frag internally
        return True, data, info
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        return False, None, info


def build_round18_eligible_response(
    response_path: str,
    *,
    feature_dir: str,
    drug_smiles_path: str,
    outdir: str,
    group_column: str = "ModelID",
    label_column: str = "Label",
    drug_column_candidates=("mapped_name", "drug_name", "DRUG_NAME"),
) -> Dict[str, Any]:
    """
    Filter response rows to those with valid omics latent + SMILES graph.

    Writes:
      data/round18_eligible_response.csv
      data/round18_removed_*.csv
      data/round18_data_eligibility_summary.json
    """
    out = Path(outdir)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    response = pd.read_csv(response_path).reset_index(drop=True)
    response["_raw_row_id"] = np.arange(len(response), dtype=int)

    drug_col = None
    for c in drug_column_candidates:
        if c in response.columns:
            drug_col = c
            break
    if drug_col is None:
        raise KeyError(f"No drug column in {response_path}; tried {drug_column_candidates}")

    meta = validate_feature_metadata(feature_dir)
    latent = load_omics_latent_dict(feature_dir)
    smiles_lookup = load_smiles_lookup(drug_smiles_path)

    # Infer expected omics dim from one latent
    sample_vec = next(iter(latent.values()))
    omics_dim = int(np.asarray(sample_vec).reshape(-1).shape[0])
    validate_feature_metadata(feature_dir, expected_dim=omics_dim)

    labels = response[label_column]
    valid_label_mask = labels.isin([0, 1, 0.0, 1.0])
    removed_label = response.loc[~valid_label_mask].copy()
    response = response.loc[valid_label_mask].copy()
    response[label_column] = response[label_column].astype(int)

    has_latent = response[group_column].astype(str).map(lambda x: x in latent)
    removed_no_latent = response.loc[~has_latent].copy()
    response = response.loc[has_latent].copy()

    response["omics_feature_key"] = response[group_column].astype(str)
    response["drug_smiles_key"] = response[drug_col].map(_normalize_drug_key)
    response["DRUG_NAME"] = response[drug_col].astype(str)

    has_smiles = response["drug_smiles_key"].map(lambda k: k in smiles_lookup)
    removed_no_smiles = response.loc[~has_smiles].copy()
    response = response.loc[has_smiles].copy()

    graph_ok = []
    graph_fail_rows = []
    graph_cache_meta = {}
    drug_graph_ok: Dict[str, bool] = {}
    for idx, row in response.iterrows():
        key = row["drug_smiles_key"]
        if key not in drug_graph_ok:
            smiles = smiles_lookup[key]
            ok, _graph, info = try_build_graph(smiles)
            graph_cache_meta[key] = info
            drug_graph_ok[key] = bool(ok)
        if drug_graph_ok[key]:
            graph_ok.append(idx)
        else:
            graph_fail_rows.append(row)

    removed_graph_failure = pd.DataFrame(graph_fail_rows)
    eligible = response.loc[graph_ok].copy().reset_index(drop=True)
    eligible["_row_id"] = np.arange(len(eligible), dtype=int)
    eligible["has_latent"] = True
    eligible["has_smiles"] = True
    eligible["graph_valid"] = True
    eligible["feature_dir"] = str(feature_dir)
    eligible["omics_dim"] = omics_dim

    paths = {
        "eligible": str(data_dir / "round18_eligible_response.csv"),
        "removed_no_latent": str(data_dir / "round18_removed_no_latent.csv"),
        "removed_no_smiles": str(data_dir / "round18_removed_no_smiles.csv"),
        "removed_graph_failure": str(data_dir / "round18_removed_graph_failure.csv"),
        "removed_invalid_label": str(data_dir / "round18_removed_invalid_label.csv"),
        "summary": str(data_dir / "round18_data_eligibility_summary.json"),
        "graph_meta": str(data_dir / "round18_drug_graph_metadata.json"),
    }
    eligible.to_csv(paths["eligible"], index=False)
    removed_no_latent.to_csv(paths["removed_no_latent"], index=False)
    removed_no_smiles.to_csv(paths["removed_no_smiles"], index=False)
    removed_graph_failure.to_csv(paths["removed_graph_failure"], index=False)
    removed_label.to_csv(paths["removed_invalid_label"], index=False)
    Path(paths["graph_meta"]).write_text(json.dumps(graph_cache_meta, indent=2), encoding="utf-8")

    summary = {
        "response_path": response_path,
        "feature_dir": feature_dir,
        "drug_smiles_path": drug_smiles_path,
        "n_raw_rows": int(len(pd.read_csv(response_path))),
        "n_eligible_rows": int(len(eligible)),
        "n_removed_no_latent": int(len(removed_no_latent)),
        "n_removed_no_smiles": int(len(removed_no_smiles)),
        "n_removed_graph_failure": int(len(removed_graph_failure)),
        "n_removed_invalid_label": int(len(removed_label)),
        "n_unique_model_ids": int(eligible[group_column].nunique()),
        "n_unique_drugs": int(eligible["DRUG_NAME"].nunique()),
        "omics_dim": omics_dim,
        "feature_metadata": {
            "n_trainable_cancer_types": meta.get("n_trainable_cancer_types"),
            "uses_legacy_28class_cache": meta.get("uses_legacy_28class_cache"),
            "prototype_class_source": meta.get("prototype_class_source"),
            "response_input_dim": meta.get("response_input_dim"),
            "prototype_feature_mode": meta.get("prototype_feature_mode"),
        },
        "paths": paths,
    }
    Path(paths["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
