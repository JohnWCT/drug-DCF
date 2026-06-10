import json
import os
from typing import Any, Dict, Set

import numpy as np
import pandas as pd
import torch


TARGET_DOMAIN_CONFIG = {
    "tcga": {
        "target_expression": "data/TCGA/pretrain_tcga.csv",
        "target_response": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv",
        "target_cancer_reference": "data/TCGA/xena_sample_info_df.csv",
    },
    "pdx": {
        "target_expression": "data/PDX/pdtc_uq1000_feature.csv",
        "target_response": "data/PDX/PDX_drug_response_from_DAPL.csv",
        "target_cancer_reference": "data/PDX/PDX_target_cancer_reference.csv",
        "target_response_label_column": "Label",
    },
}


def to_scalar(v: Any) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def json_safe(data: Any):
    if isinstance(data, dict):
        return {str(k): json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [json_safe(v) for v in data]
    if isinstance(data, tuple):
        return [json_safe(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.bool_,)):
        return bool(data)
    if isinstance(data, torch.Tensor):
        if data.numel() == 1:
            return float(data.detach().cpu().item())
        return data.detach().cpu().tolist()
    return data


def is_tcga_sample(sample_id: str) -> bool:
    return isinstance(sample_id, str) and sample_id.startswith("TCGA-")


def tcga_three_segment_key(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return str(sample_id)


def tcga_sample_priority(sample_id: str):
    parts = str(sample_id).split("-")
    if len(parts) < 4:
        return (0, 0)
    seg = parts[3].upper()
    num_str = "".join(c for c in seg if c.isdigit())
    numeric = int(num_str) if num_str else 999
    letter_rank = 0
    if "A" in seg:
        letter_rank = 0
    elif "B" in seg:
        letter_rank = 1
    return (numeric, letter_rank)


def deduplicate_tcga_latent_dict(latent_dict: Dict[str, Any]):
    result = {}
    tcga_groups = {}
    for sample_id, latent in latent_dict.items():
        if not is_tcga_sample(sample_id):
            result[sample_id] = latent
            continue
        key = tcga_three_segment_key(sample_id)
        priority = tcga_sample_priority(sample_id)
        if key not in tcga_groups:
            tcga_groups[key] = []
        tcga_groups[key].append((priority, sample_id))
    for key, candidates in tcga_groups.items():
        best = min(candidates, key=lambda x: x[0])
        result[key] = latent_dict[best[1]]
    return result


def load_overlap_patient_ids(overlap_csv: str):
    if not overlap_csv or (not os.path.exists(overlap_csv)):
        return set()
    df = pd.read_csv(overlap_csv)
    if "Patient_id" not in df.columns:
        return set()
    vals = df["Patient_id"].dropna().astype(str).tolist()
    return {tcga_three_segment_key(v) for v in vals}


def prepare_training_target_csv(
    target_path: str,
    overlap_csv: str,
    out_dir: str,
    tmp_suffix: str = "",
):
    overlap_ids = load_overlap_patient_ids(overlap_csv)
    if not overlap_ids:
        return target_path, 0
    df = pd.read_csv(target_path, index_col=0)
    idx = df.index.astype(str)
    keep_mask = [tcga_three_segment_key(v) not in overlap_ids for v in idx]
    filtered_df = df.loc[keep_mask]
    removed = int(len(df) - len(filtered_df))
    if removed == 0:
        return target_path, 0
    if len(filtered_df) == 0:
        raise ValueError(
            f"After overlap filtering, TCGA training samples become 0 "
            f"(original={len(df)}, removed={removed}). "
            f"Please provide a non-overlap TCGA training set or adjust overlap list."
        )
    suffix = f"_{tmp_suffix}" if tmp_suffix else ""
    train_target_path = os.path.join(out_dir, f"_tmp_target_for_training{suffix}.csv")
    filtered_df.to_csv(train_target_path)
    return train_target_path, removed


def compute_class_weights(labels, device):
    labels = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights * (len(class_counts) / weights.sum())
    return torch.from_numpy(weights).float().to(device)


DEFAULT_CANCER_TYPE_EXCLUDE_CONFIG = os.path.join("config", "pretrain_cancer_type_exclude.json")


def normalize_cancer_type_token(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower().replace("&", "and")


def load_cancer_type_exclude_set(config_path: str = DEFAULT_CANCER_TYPE_EXCLUDE_CONFIG) -> Set[str]:
    """Load normalized cancer_type tokens to exclude from pretrain (e.g. 'na')."""
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("exclude_from_training", ["na"])
    if not items:
        raise ValueError(f"exclude_from_training is empty in {config_path}")
    return {normalize_cancer_type_token(x) for x in items}


def is_trainable_cancer_type(value: Any, exclude_set: Set[str]) -> bool:
    """True if value is non-empty and not listed in exclude_from_training config."""
    token = normalize_cancer_type_token(value)
    if not token:
        return False
    return token not in exclude_set


def filter_trainable_cancer_types(series: pd.Series, exclude_set: Set[str]) -> pd.Series:
    return series.map(lambda v: is_trainable_cancer_type(v, exclude_set))
