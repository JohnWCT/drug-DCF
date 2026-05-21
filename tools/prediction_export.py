"""
Per-sample prediction tables (CCLE test / TCGA eval).

Aligned with C_prototypical.py outputs:
  ccle_test_predictions.csv
  tcga_eval_predictions.csv

Core columns: sample_id, drug_id, domain, ground_truth (GT), confidence.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tools.dataprocess import safemakedirs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_THRESHOLD = 0.5


def logits_to_confidence(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _metadata_from_dataset_row(dataset, idx: int) -> Dict[str, Any]:
    row = dataset.df.iloc[idx]
    if hasattr(dataset, "col_map"):
        return {
            "sample_id": str(row[dataset.col_map["sample"]]),
            "drug_id": str(row[dataset.col_map["drug"]]),
            "ground_truth": float(row[dataset.col_map["label"]]),
        }
    if "DepMap_ID" in row:
        sample_id = str(row["DepMap_ID"])
    elif "ModelID" in row:
        sample_id = str(row["ModelID"])
    else:
        sample_id = str(row.iloc[0])
    if "drug_name" in row:
        drug_id = str(row["drug_name"])
    elif "DRUG_NAME" in row:
        drug_id = str(row["DRUG_NAME"])
    else:
        drug_id = str(row.iloc[1])
    if "Label" in row:
        gt = float(row["Label"])
    elif "Class" in row:
        gt = float(row["Class"])
    else:
        gt = float(row.iloc[2])
    return {"sample_id": sample_id, "drug_id": drug_id, "ground_truth": gt}


def _encode_gene_batch(model_components, batch_gene):
    encoder = model_components.get("encoder")
    if encoder is None:
        if isinstance(batch_gene, list):
            batch_gene = torch.stack([g if isinstance(g, torch.Tensor) else torch.as_tensor(g) for g in batch_gene])
        return batch_gene.to(device)

    if isinstance(batch_gene, list):
        batch_gene = torch.stack([g if isinstance(g, torch.Tensor) else torch.as_tensor(g) for g in batch_gene])
    batch_gene = batch_gene.to(device)
    encoder.eval()
    encoder_type = model_components.get("encoder_type", "vae")
    with torch.no_grad():
        if encoder_type == "vae":
            _, z, _, _ = encoder(batch_gene)
            return z
        return encoder(batch_gene)


def _forward_batch(model_components, batch_gene, batch_drug, model_params=None):
    drug_model = model_components["drug_model"]
    classifier = model_components["classifier"]

    batch_gene = _encode_gene_batch(model_components, batch_gene)

    if isinstance(batch_drug, torch.Tensor):
        drug_emb = batch_drug.to(device)
    elif isinstance(batch_drug, list):
        if batch_drug and isinstance(batch_drug[0], torch.Tensor):
            drug_emb = torch.stack(batch_drug).to(device)
        else:
            from torch_geometric.data import Batch

            drug_emb = drug_model(Batch.from_data_list(batch_drug).to(device))
    else:
        drug_emb = drug_model(batch_drug.to(device))

    combined = torch.cat((batch_gene, drug_emb), dim=1)
    logits = classifier(combined).view(-1)
    return logits.detach().cpu().numpy()


def collect_ccle_predictions(
    model_components,
    dataset,
    model_params=None,
    domain: str = "CCLE",
    fold_id: Optional[int] = None,
    batch_size: int = 2048,
    collate_fn=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Run inference on a DrugResponseDataset and return per-sample prediction rows."""
    if dataset is None or len(dataset) == 0:
        return pd.DataFrame()

    drug_model = model_components.get("drug_model")
    if drug_model is not None:
        drug_model.eval()
    model_components["classifier"].eval()

    if collate_fn is None:
        def collate_fn(batch):
            gene_list, drug_list, target_list = zip(*batch)
            return list(gene_list), list(drug_list), list(target_list)

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=False,
        collate_fn=collate_fn,
    )

    rows: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                batch_gene, batch_drug, batch_target, _weights = batch
            else:
                batch_gene, batch_drug, batch_target = batch

            logits = _forward_batch(model_components, batch_gene, batch_drug, model_params)
            conf = logits_to_confidence(logits)
            n = len(conf)

            for i in range(n):
                meta = _metadata_from_dataset_row(dataset, offset + i)
                row = {
                    "sample_id": meta["sample_id"],
                    "drug_id": meta["drug_id"],
                    "domain": domain,
                    "ground_truth": meta["ground_truth"],
                    "confidence": float(conf[i]),
                    "prediction_binary": int(conf[i] >= threshold),
                    "prediction": float(conf[i]),
                    "threshold": threshold,
                }
                if fold_id is not None:
                    row["fold"] = int(fold_id)
                rows.append(row)
            offset += n

    return pd.DataFrame(rows)


def build_tcga_prediction_rows(
    patient_ids: List[str],
    drug_name: str,
    ground_truth: np.ndarray,
    confidence: np.ndarray,
    fold_id: Optional[int] = None,
    tcga_source: str = "TCGA1",
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict[str, Any]]:
    rows = []
    for pid, gt, conf in zip(patient_ids, ground_truth, confidence):
        rows.append({
            "sample_id": str(pid),
            "drug_id": str(drug_name),
            "domain": "TCGA",
            "ground_truth": float(gt),
            "confidence": float(conf),
            "prediction_binary": int(float(conf) >= threshold),
            "prediction": float(conf),
            "threshold": threshold,
            "original_drug_name": str(drug_name),
            "tcga_source": tcga_source,
            **({"fold": int(fold_id)} if fold_id is not None else {}),
        })
    return rows


def predictions_from_tcga_inference_result(
    tcga_results: Dict,
    tcga_source: str = "TCGA1",
) -> pd.DataFrame:
    rows = tcga_results.get("Sample_Predictions", []) if isinstance(tcga_results, dict) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "tcga_source" not in df.columns:
        df["tcga_source"] = tcga_source
    return df


def save_prediction_tables(
    output_dir: str,
    ccle_test_df: Optional[pd.DataFrame] = None,
    tcga_eval_df: Optional[pd.DataFrame] = None,
    tcga_eval_extra_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """Save CCLE / TCGA per-sample prediction CSVs under output_dir."""
    safemakedirs(output_dir)
    saved = {}
    if ccle_test_df is not None and not ccle_test_df.empty:
        path = os.path.join(output_dir, "ccle_test_predictions.csv")
        ccle_test_df.to_csv(path, index=False)
        saved["ccle_test_predictions"] = path
        print(f"[predictions] saved {path} ({len(ccle_test_df)} rows)")
    if tcga_eval_df is not None and not tcga_eval_df.empty:
        path = os.path.join(output_dir, "tcga_eval_predictions.csv")
        tcga_eval_df.to_csv(path, index=False)
        saved["tcga_eval_predictions"] = path
        print(f"[predictions] saved {path} ({len(tcga_eval_df)} rows)")
    if tcga_eval_extra_df is not None and not tcga_eval_extra_df.empty:
        path = os.path.join(output_dir, "tcga_eval_predictions_TCGA2.csv")
        tcga_eval_extra_df.to_csv(path, index=False)
        saved["tcga_eval_predictions_TCGA2"] = path
        print(f"[predictions] saved {path} ({len(tcga_eval_extra_df)} rows)")
    return saved


def aggregate_fold_prediction_dfs(fold_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    valid = [df for df in fold_dfs if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)
