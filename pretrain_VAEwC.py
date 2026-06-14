"""
VAEwC pretraining + GAN alignment pipeline.

Usage example:
python pretrain_VAEwC.py \
  --config config/params_from_model_select_fulltest_A_loss_earlystop.json \
  --outfolder result/pretrain_vaewc \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv
"""

import os
import re
import json
import copy
import pickle
import argparse
import fcntl
import itertools
import time
from collections import defaultdict
from itertools import chain, cycle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import torch
import torch.nn as nn

from tools.dataprocess import safemakedirs, append_csv_log
from tools.pretrain_tsne import plot_latent_tsne_dual
from tools.model_opt import VAE, Discriminator, MLP, vaeloss, init_weights, ortho_loss, compute_gradient_penalty
from tools.pretrain_common import (
    TARGET_DOMAIN_CONFIG,
    to_scalar as _to_scalar,
    json_safe as _json_safe,
    tcga_three_segment_key,
    deduplicate_tcga_latent_dict,
    prepare_training_target_csv as _prepare_training_target_csv,
    compute_class_weights as _compute_class_weights,
)
from tools.proto_infonce import compute_prototype_infonce, default_proto_metrics
from tools.classwise_alignment import compute_classwise_mmd, compute_classwise_prototype_gap
from tools.tumor_geometry import compute_tumor_topology_loss
from tools.tumor_subspace import (
    alignment_discriminator_input,
    classifier_input_dim,
    discriminator_input_dim,
    resolve_subspace_training_params,
    select_latent_view,
    split_tumor_transfer_latent,
    compute_subspace_orthogonality_loss,
)
from tools.tumor_supcon import compute_within_domain_supcon_loss
from tools.tumor_vicreg import compute_vicreg_var_cov_loss
from tools.pretrain_proto_schedule import (
    get_lambda_proto_eff,
    get_lambda_cmmd_eff,
    get_lambda_class_gap_eff,
    get_lambda_tumor_topology_eff,
    get_lambda_tumor_supcon_eff,
    get_lambda_tumor_var_eff,
    get_lambda_tumor_cov_eff,
    get_lambda_subspace_ortho_eff,
    resolve_proto_training_params,
    resolve_class_gap_training_params,
    resolve_cmmd_training_params,
    resolve_tumor_topology_training_params,
    resolve_tumor_supcon_training_params,
    resolve_tumor_vicreg_training_params,
    compute_proto_checkpoint_guard,
    post_proto_checkpoint_min_epoch,
)
from tools.proto_structure_metrics import compute_proto_structure_metrics


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
plt.switch_backend("Agg")
print("use device:", device)

# Switchable backbone for VAEwC/AEwC shared pipeline.
MODEL_BACKBONE = VAE
MODEL_TYPE_NAME = "VAE"
DEFAULT_SOURCE_CSV = "data/pretrain_ccle.csv"
DEFAULT_CCLE_INFO_CSV = os.path.join("data", "ccle_sample_info_df.csv")
PRETRAIN_MODEL_SELECT_FILENAME = "pretrain_model_select.csv"
# Learning curve display control:
# skip first N epochs on x-axis when plotting.
CURVE_SKIP_INITIAL_EPOCHS = 1


def smooth_rampup(epoch: int, start_epoch: int, end_epoch: int, max_value: float) -> float:
    """Smooth cubic ramp from 0 to max_value between start_epoch and end_epoch."""
    if max_value <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return float(max_value) if epoch >= start_epoch else 0.0
    if epoch >= end_epoch:
        return float(max_value)
    phase = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(max_value) * (phase ** 3)


def get_lambda_cls_eff(epoch: int, param: dict) -> float:
    """Effective classifier weight for pretraining (staged warm-up)."""
    lambda_cls = float(param.get("lambda_cls", 1.0))
    if lambda_cls <= 0:
        return 0.0
    cls_start_epoch = int(param.get("cls_start_epoch", 1))
    cls_full_epoch = int(param.get("cls_full_epoch", cls_start_epoch))
    return smooth_rampup(epoch, cls_start_epoch, cls_full_epoch, lambda_cls)


def _mean_loss_logs(logs):
    """Average per-step loss dicts; skip non-numeric metadata (e.g. proto_mode strings)."""
    mean = defaultdict(float)
    if not logs:
        return mean
    counts = defaultdict(int)
    for log in logs:
        for k, v in log.items():
            if isinstance(v, (str, bool)):
                continue
            try:
                mean[k] += _to_scalar(v)
                counts[k] += 1
            except (TypeError, ValueError):
                continue
    for k in mean:
        mean[k] /= counts[k]
    return mean


def resolve_gan_training_params(param: dict, gan_lr: float, lambda_cls: float) -> dict:
    """Resolve GAN-stage schedule/config with backward-compatible defaults."""
    return {
        "gan_gen_update_interval": max(1, int(param.get("gan_gen_update_interval", 5))),
        "gan_cls_update_every_step": bool(param.get("gan_cls_update_every_step", True)),
        "gan_cls_learning_rate": float(param.get("gan_cls_learning_rate", gan_lr)),
        "gan_lambda_cls": float(param.get("gan_lambda_cls", lambda_cls)),
        "gan_gp_weight": float(param.get("gan_gp_weight", 10.0)),
    }


def resolve_schedule_hyperparams(param: dict) -> dict:
    """Flatten schedule/GAN hyperparameters with resolved defaults for CSV export."""
    gan_lr = float(param.get("gan_learning_rate", 0.0005))
    lambda_cls = float(param.get("lambda_cls", 1.0))
    cls_start_epoch = int(param.get("cls_start_epoch", 1))
    cls_full_epoch = int(param.get("cls_full_epoch", cls_start_epoch))
    gan_cfg = resolve_gan_training_params(param, gan_lr, lambda_cls)
    proto_cfg = resolve_proto_training_params(param)
    cmmd_cfg = resolve_cmmd_training_params(param)
    return {
        "cls_start_epoch": cls_start_epoch,
        "cls_full_epoch": cls_full_epoch,
        "gan_gen_update_interval": gan_cfg["gan_gen_update_interval"],
        "gan_cls_update_every_step": gan_cfg["gan_cls_update_every_step"],
        "gan_cls_learning_rate": gan_cfg["gan_cls_learning_rate"],
        "gan_lambda_cls": gan_cfg["gan_lambda_cls"],
        "gan_gp_weight": gan_cfg["gan_gp_weight"],
        **proto_cfg,
        **cmmd_cfg,
    }


def build_experiment_summary_row(param_dict: dict, exp_name: str, metrics: dict) -> dict:
    """Build one flat CSV row with scores and all resolved training hyperparameters."""
    sched = resolve_schedule_hyperparams(param_dict)
    row = {
        "ID": exp_name,
        "NO": "",
        "model_type": MODEL_TYPE_NAME,
        "pretrain_epochs": param_dict.get("pretrain_num_epochs"),
        "train_epochs": param_dict.get("train_num_epochs"),
        "pretrain_lr": param_dict.get("pretrain_learning_rate"),
        "train_lr": param_dict.get("gan_learning_rate"),
        "dropout": param_dict.get("dropout_rate"),
        "latent_size": param_dict.get("latent_size", 32),
        "encoder_dims": str(param_dict.get("encoder_dims")),
        "lambda_cls": param_dict.get("lambda_cls"),
        "use_class_weight": param_dict.get("use_class_weight", False),
        "cls_start_epoch": sched["cls_start_epoch"],
        "cls_full_epoch": sched["cls_full_epoch"],
        "gan_gen_update_interval": sched["gan_gen_update_interval"],
        "gan_cls_update_every_step": sched["gan_cls_update_every_step"],
        "gan_cls_learning_rate": sched["gan_cls_learning_rate"],
        "gan_lambda_cls": sched["gan_lambda_cls"],
        "gan_gp_weight": sched["gan_gp_weight"],
        "FID_AfterGAN": metrics["fid"],
        "MMD_AfterGAN": metrics["mmd"],
        "Wasserstein_AfterGAN": metrics["wasserstein"],
        "best_gan_epoch": metrics["best_gan_epoch"],
        "best_gan_loss": metrics["best_gan_loss"],
        "fid": metrics["fid"],
        "mmd": metrics["mmd"],
        "wasserstein": metrics["wasserstein"],
        "kmeans_k": metrics.get("kmeans_k"),
        "kmeans_ari": metrics.get("kmeans_ari"),
        "kmeans_nmi": metrics.get("kmeans_nmi"),
        "kmeans_silhouette": metrics.get("kmeans_silhouette"),
        "kmeans_calinski_harabasz": metrics.get("kmeans_calinski_harabasz"),
        "kmeans_davies_bouldin": metrics.get("kmeans_davies_bouldin"),
        "result_folder": exp_name,
        "lambda_proto": sched.get("lambda_proto", 0.0),
        "proto_temperature": sched.get("proto_temperature", 0.2),
        "proto_start_epoch": sched.get("proto_start_epoch", 1),
        "proto_full_epoch": sched.get("proto_full_epoch", 1),
        "lambda_adv": sched.get("lambda_adv", 1.0),
        "proto_mode": sched.get("proto_mode", "combined"),
        "proto_direction": sched.get("proto_direction", "symmetric"),
        "proto_detach": sched.get("proto_detach", True),
        "proto_min_samples_per_domain": sched.get("proto_min_samples_per_domain", 1),
        "lambda_cmmd": sched.get("lambda_cmmd", 0.0),
        "best_proto_loss": metrics.get("best_proto_loss"),
        "best_proto_margin": metrics.get("best_proto_margin"),
        "best_proto_acc": metrics.get("best_proto_acc"),
    }
    return row


class PrimaryClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [64], dop: float = 0.1, act_fn=nn.ReLU):
        super().__init__()
        self.net = MLP(input_dim=input_dim, output_dim=num_classes, hidden_dims=hidden_dims, dop=dop, act_fn=act_fn)

    def forward(self, x):
        return self.net(x)


def _cap_batch_size(batch_size: int, n_samples: int) -> int:
    """Ensure drop_last=True DataLoaders retain at least one batch."""
    n_samples = int(n_samples)
    batch_size = int(batch_size)
    if n_samples < 2:
        return max(1, batch_size)
    if batch_size >= n_samples:
        capped = max(1, n_samples // 2)
        print(
            f"[batch_size] capped {batch_size} -> {capped} "
            f"(dataset rows={n_samples}, drop_last=True requires batch < n)"
        )
        return capped
    return batch_size


def _load_labeled_data_patient_aware(
    ccle_path,
    xena_path,
    batch_size=128,
    use_class_weight=False,
    target_domain="tcga",
    target_cancer_reference_path=None,
    ccle_info_path=None,
):
    """Load labeled data; expects pre-cleaned expression CSVs (see tools/clean_pretrain_inputs_by_cancer_type.py)."""
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    xena_df = pd.read_csv(xena_path, index_col=0)
    ccle_df.index = ccle_df.index.astype(str)
    xena_df.index = xena_df.index.astype(str)
    common_cols = [c for c in ccle_df.columns if c in set(xena_df.columns)]
    if len(common_cols) == 0:
        raise ValueError(
            f"No overlapping feature columns between source ({ccle_path}) "
            f"and target ({xena_path})."
        )
    ccle_df = ccle_df.loc[:, common_cols]
    xena_df = xena_df.loc[:, common_cols]

    ccle_info_path = ccle_info_path or DEFAULT_CCLE_INFO_CSV
    ccle_info = pd.read_csv(ccle_info_path, index_col=0, header=0)
    ccle_info.index = ccle_info.index.astype(str)
    if "cancer_type" not in ccle_info.columns:
        raise ValueError(
            f"Missing cancer_type in {ccle_info_path}. "
            f"Run: python tools/add_xena_cancer_type_column.py --ccle-only"
        )
    ccle_info["primary_disease"] = ccle_info["cancer_type"].astype(str).str.strip()

    target_domain = str(target_domain).lower()
    if target_domain == "tcga":
        if not target_cancer_reference_path:
            target_cancer_reference_path = TARGET_DOMAIN_CONFIG["tcga"]["target_cancer_reference"]
        xena_info = pd.read_csv(target_cancer_reference_path, index_col=0, header=0)
        xena_info.index = xena_info.index.astype(str)
        if "cancer_type" not in xena_info.columns:
            raise ValueError(
                f"Missing cancer_type in {target_cancer_reference_path}. "
                f"Run: python tools/add_xena_cancer_type_column.py --tcga-only"
            )
        xena_info["_primary_disease"] = xena_info["cancer_type"].astype(str).str.strip()
        xena_info["patient_id"] = xena_info.index.map(tcga_three_segment_key)
        xena_patient_info = (
            xena_info.dropna(subset=["_primary_disease"])
            .sort_index()
            .groupby("patient_id")
            .first()
        )
    elif target_domain == "pdx":
        if not target_cancer_reference_path:
            target_cancer_reference_path = TARGET_DOMAIN_CONFIG["pdx"]["target_cancer_reference"]
        xena_info = pd.read_csv(target_cancer_reference_path)
        if "Model" in xena_info.columns and ("cancer_type" in xena_info.columns or "cancerType" in xena_info.columns):
            cancer_col = "cancer_type" if "cancer_type" in xena_info.columns else "cancerType"
            xena_info["Model"] = xena_info["Model"].astype(str)
            xena_info["_primary_disease"] = xena_info[cancer_col].astype(str).str.strip()
            xena_patient_info = (
                xena_info.dropna(subset=["_primary_disease"])
                .sort_values("Model")
                .drop_duplicates(subset=["Model"], keep="first")
                .set_index("Model")
            )
        else:
            # New PDX setting: all samples are breast cancer and response table
            # may only contain Sample_id without cancerType annotations.
            sample_col = "Sample_id" if "Sample_id" in xena_info.columns else xena_info.columns[0]
            xena_info[sample_col] = xena_info[sample_col].astype(str)
            xena_patient_info = (
                xena_info[[sample_col]]
                .drop_duplicates(subset=[sample_col], keep="first")
                .set_index(sample_col)
            )
            xena_patient_info["_primary_disease"] = "Breast Cancer"
    else:
        raise ValueError(f"Unsupported target_domain={target_domain}. Use tcga or pdx.")
    ccle_info.index = ccle_info.index.astype(str)
    valid_ccle_ids = ccle_df.index.intersection(ccle_info.index)
    ccle_info = ccle_info.loc[valid_ccle_ids]
    ccle_df = ccle_df.loc[valid_ccle_ids]
    ccle_info = ccle_info.dropna(subset=["primary_disease"])
    ccle_df = ccle_df.loc[ccle_info.index]
    disease_count = ccle_info.primary_disease.value_counts()
    ccle_keep = disease_count[disease_count >= 10].index
    ccle_info = ccle_info[ccle_info.primary_disease.isin(ccle_keep)]
    ccle_df = ccle_df.loc[ccle_info.index]
    # Align TCGA expression (sample-level IDs like TCGA-XX-XXXX-07) to
    # patient-level IDs (TCGA-XX-XXXX) used by cancer reference mapping.
    xena_df = xena_df.copy()
    xena_df["patient_id"] = xena_df.index.map(tcga_three_segment_key)
    xena_df = (
        xena_df.sort_index()
        .groupby("patient_id", as_index=True)
        .first()
    )
    valid_tcga_ids = xena_df.index.intersection(xena_patient_info.index)
    xena_df = xena_df.loc[valid_tcga_ids]
    xena_labels = xena_patient_info.loc[valid_tcga_ids, "_primary_disease"]
    common_labels = sorted(set(ccle_info.primary_disease.unique()) & set(xena_labels.unique()))
    ccle_mask = ccle_info.primary_disease.isin(common_labels)
    xena_mask = xena_labels.isin(common_labels)
    ccle_df = ccle_df.loc[ccle_mask]
    ccle_labels = ccle_info.loc[ccle_mask, "primary_disease"]
    xena_df = xena_df.loc[xena_mask]
    xena_labels = xena_labels.loc[xena_mask]
    if len(ccle_df) == 0 or len(xena_df) == 0:
        raise ValueError("No valid labeled samples after filtering. Please verify sample IDs and disease mapping.")
    from sklearn.model_selection import train_test_split
    ccle_train, ccle_test, ccle_train_y, ccle_test_y = train_test_split(
        ccle_df, ccle_labels, test_size=0.2, stratify=ccle_labels, random_state=42
    )
    xena_train, xena_test, xena_train_y, xena_test_y = train_test_split(
        xena_df, xena_labels, test_size=0.2, stratify=xena_labels, random_state=42
    )
    label_map = {d: i for i, d in enumerate(common_labels)}
    mapping_int2str = {i: d for d, i in label_map.items()}
    ccle_train_label_int = np.array([label_map[x] for x in ccle_train_y], dtype=np.int64)
    ccle_test_label_int = np.array([label_map[x] for x in ccle_test_y], dtype=np.int64)
    xena_train_label_int = np.array([label_map[x] for x in xena_train_y], dtype=np.int64)
    xena_test_label_int = np.array([label_map[x] for x in xena_test_y], dtype=np.int64)
    source_train_tensor = torch.from_numpy(ccle_train.values).float().to(device)
    source_test_tensor = torch.from_numpy(ccle_test.values).float().to(device)
    target_train_tensor = torch.from_numpy(xena_train.values).float().to(device)
    target_test_tensor = torch.from_numpy(xena_test.values).float().to(device)
    source_train_label_tensor = torch.from_numpy(ccle_train_label_int).to(device)
    source_test_label_tensor = torch.from_numpy(ccle_test_label_int).to(device)
    target_train_label_tensor = torch.from_numpy(xena_train_label_int).to(device)
    target_test_label_tensor = torch.from_numpy(xena_test_label_int).to(device)
    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(source_train_tensor, source_train_label_tensor),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(target_train_tensor, target_train_label_tensor),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    if use_class_weight:
        source_weights = _compute_class_weights(ccle_train_label_int, device)
        target_weights = _compute_class_weights(xena_train_label_int, device)
        sourcedata = (source_loader, source_test_tensor, source_test_label_tensor, source_weights, mapping_int2str)
        targetdata = (target_loader, target_test_tensor, target_test_label_tensor, target_weights, mapping_int2str)
    else:
        sourcedata = (source_loader, source_test_tensor, source_test_label_tensor, mapping_int2str)
        targetdata = (target_loader, target_test_tensor, target_test_label_tensor, mapping_int2str)
    return sourcedata, targetdata


def _next_experiment_dir(parent_folder: str) -> Tuple[str, str]:
    safemakedirs(parent_folder)
    lock_path = os.path.join(parent_folder, ".exp_id.lock")
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        lock_file.seek(0)
        existing = []
        for name in os.listdir(parent_folder):
            m = re.fullmatch(r"exp_(\d+)", name)
            if m:
                existing.append(int(m.group(1)))
        next_id = (max(existing) + 1) if existing else 1
        exp_name = f"exp_{next_id:03d}"
        exp_dir = os.path.join(parent_folder, exp_name)
        safemakedirs(exp_dir)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(next_id))
        lock_file.flush()
    return exp_name, exp_dir


def _calculate_mmd(source_latent: np.ndarray, target_latent: np.ndarray, gamma=None) -> float:
    if source_latent.shape[0] > 1000:
        source_latent = source_latent[np.random.choice(source_latent.shape[0], 1000, replace=False)]
    if target_latent.shape[0] > 1000:
        target_latent = target_latent[np.random.choice(target_latent.shape[0], 1000, replace=False)]
    if gamma is None:
        gamma = 1.0 / source_latent.shape[1]
    xx = np.exp(-gamma * cdist(source_latent, source_latent, "sqeuclidean"))
    yy = np.exp(-gamma * cdist(target_latent, target_latent, "sqeuclidean"))
    xy = np.exp(-gamma * cdist(source_latent, target_latent, "sqeuclidean"))
    return float(max(0.0, xx.mean() + yy.mean() - 2 * xy.mean()))


def _calculate_wasserstein(source_latent: np.ndarray, target_latent: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(source_latent, axis=0) - np.mean(target_latent, axis=0)))


def _sanitize_latent_for_fid(latent: np.ndarray, name: str = "latent") -> np.ndarray:
    """Drop rows containing NaN / Inf so downstream covariance / sqrtm stays finite.

    Returning an empty array signals the caller that FID cannot be computed
    from this side and should fall back to ``np.inf``.
    """
    if latent is None:
        return None
    arr = np.asarray(latent, dtype=np.float64)
    if arr.ndim != 2:
        return arr
    finite_mask = np.isfinite(arr).all(axis=1)
    dropped = int(arr.shape[0] - finite_mask.sum())
    if dropped > 0:
        print(
            f"[fid] warning: dropped {dropped}/{arr.shape[0]} rows with "
            f"NaN/Inf from {name} before FID computation"
        )
    return arr[finite_mask]


def _compute_fid(source_latent: np.ndarray, target_latent: np.ndarray = None) -> float:
    """Compute Frechet Distance between two latent distributions.

    Robust against NaN/Inf latents and sqrtm convergence failures: any such
    degenerate case returns ``float('inf')`` so callers using it as an
    early-stopping metric never crash and never mistake a failed epoch for
    an improvement (``inf`` will never beat the running best).
    """
    try:
        source_clean = _sanitize_latent_for_fid(source_latent, name="source_latent")
        if source_clean is None or source_clean.shape[0] < 2:
            print("[fid] warning: source latent has <2 finite rows; returning inf")
            return float("inf")

        mu = np.mean(source_clean, axis=0)
        sigma = np.cov(source_clean, rowvar=False)

        if target_latent is None:
            prior = np.random.randn(*source_clean.shape)
            mu2 = np.mean(prior, axis=0)
            sigma2 = np.cov(prior, rowvar=False)
        else:
            target_clean = _sanitize_latent_for_fid(target_latent, name="target_latent")
            if target_clean is None or target_clean.shape[0] < 2:
                print("[fid] warning: target latent has <2 finite rows; returning inf")
                return float("inf")
            mu2 = np.mean(target_clean, axis=0)
            sigma2 = np.cov(target_clean, rowvar=False)

        if not (np.isfinite(mu).all() and np.isfinite(sigma).all()
                and np.isfinite(mu2).all() and np.isfinite(sigma2).all()):
            print("[fid] warning: non-finite mu/sigma after sanitisation; returning inf")
            return float("inf")

        from tools.metrics import calculate_frechet_distance
        value = float(calculate_frechet_distance(mu, sigma, mu2, sigma2))
        if not np.isfinite(value):
            return float("inf")
        return value
    except Exception as err:  # pragma: no cover - defensive catch-all
        print(f"[fid] warning: FID computation failed ({err}); returning inf")
        return float("inf")


def _kmeans_combined_metrics(
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    n_clusters: int,
):
    metrics = {
        "kmeans_k": np.nan,
        "kmeans_ari": np.nan,
        "kmeans_nmi": np.nan,
        "kmeans_silhouette": np.nan,
        "kmeans_calinski_harabasz": np.nan,
        "kmeans_davies_bouldin": np.nan,
    }
    if source_latent is None or target_latent is None:
        return metrics
    if len(source_latent) < 2 or len(target_latent) < 2:
        return metrics
    latent = np.vstack([source_latent, target_latent])
    labels = np.concatenate([
        np.asarray(source_labels, dtype=np.int64),
        np.asarray(target_labels, dtype=np.int64),
    ])
    if len(latent) < 3:
        return metrics
    k = int(max(2, min(n_clusters, len(np.unique(labels)), len(latent) - 1)))
    if k < 2:
        return metrics
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(latent)
    ari_raw = float(adjusted_rand_score(labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(labels, cluster_labels))
    metrics["kmeans_k"] = int(k)
    metrics["kmeans_ari"] = ari_raw
    metrics["kmeans_nmi"] = max(0.0, min(1.0, nmi))
    try:
        metrics["kmeans_silhouette"] = float(silhouette_score(latent, cluster_labels))
    except Exception:
        metrics["kmeans_silhouette"] = np.nan
    try:
        metrics["kmeans_calinski_harabasz"] = float(calinski_harabasz_score(latent, cluster_labels))
    except Exception:
        metrics["kmeans_calinski_harabasz"] = np.nan
    try:
        metrics["kmeans_davies_bouldin"] = float(davies_bouldin_score(latent, cluster_labels))
    except Exception:
        metrics["kmeans_davies_bouldin"] = np.nan
    return metrics


def _plot_gan_tsne(source_z, target_z, source_labels, target_labels, mapping_int2str, save_path):
    """Dual-panel t-SNE: A=domain, B=cancer type (see tools.pretrain_tsne)."""
    plot_latent_tsne_dual(
        source_z,
        target_z,
        source_labels,
        target_labels,
        mapping_int2str,
        save_path,
        suptitle="GAN Best Latent t-SNE (Test Split)",
    )


def _encode_latent_dict(model, feature_df: pd.DataFrame, batch_size=512):
    model.eval()
    latents = {}
    ids = feature_df.index.astype(str).tolist()
    x = torch.from_numpy(feature_df.values).float().to(device)
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(len(ids), start + batch_size)
            _, z, _, _ = model(x[start:end])
            z_np = z.detach().cpu().numpy()
            for i, sid in enumerate(ids[start:end]):
                latents[sid] = z_np[i].tolist()
    return latents


def train_discrim(
    s_batch,
    t_batch,
    shared_encoder,
    sencoder,
    tencoder,
    discrim,
    optimizer,
    scheduler,
    gan_gp_weight: float = 10.0,
    subspace_cfg=None,
):
    """WGAN-GP discriminator step: encoders frozen, critic learns from latent scores."""
    subspace_cfg = subspace_cfg or resolve_subspace_training_params({})
    loss_log = defaultdict(float)
    discrim.zero_grad()
    sencoder.eval()
    tencoder.eval()
    shared_encoder.eval()
    discrim.train()
    optimizer.zero_grad()
    with torch.no_grad():
        _, pzs, _, _ = sencoder(s_batch)
        _, pzt, _, _ = tencoder(t_batch)
        _, zs, _, _ = shared_encoder(s_batch)
        _, zt, _, _ = shared_encoder(t_batch)
        s = alignment_discriminator_input(zs, pzs, subspace_cfg)
        t = alignment_discriminator_input(zt, pzt, subspace_cfg)
    d_s = discrim(s)
    d_t = discrim(t)
    d_loss = torch.mean(d_t) - torch.mean(d_s)
    g_p = compute_gradient_penalty(critic=discrim, real_samples=s, fake_samples=t, device=device)
    total_loss = d_loss + gan_gp_weight * g_p
    loss_log.update({
        "discrim_loss": d_loss,
        "g_p": g_p,
        "discrim_total_loss": total_loss,
        "d_source_score": torch.mean(d_s),
        "d_target_score": torch.mean(d_t),
    })
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    discrim.eval()
    return loss_log


def train_classifier_step(
    s_batch,
    t_batch,
    s_labels,
    t_labels,
    shared_encoder,
    classifier,
    optimizer,
    scheduler,
    source_weights=None,
    target_weights=None,
    use_class_weight: bool = False,
    subspace_cfg=None,
):
    """Classifier-only GAN step: shared encoder frozen, classifier adapts to current latents."""
    subspace_cfg = subspace_cfg or resolve_subspace_training_params({})
    cls_view = subspace_cfg.get("classifier_latent_view", "shared")
    loss_log = defaultdict(float)
    classifier.zero_grad()
    shared_encoder.eval()
    classifier.train()
    optimizer.zero_grad()
    _, ccle_z, _, _ = shared_encoder(s_batch)
    _, tcga_z, _, _ = shared_encoder(t_batch)
    ccle_z = select_latent_view(ccle_z, cls_view, subspace_cfg)
    tcga_z = select_latent_view(tcga_z, cls_view, subspace_cfg)
    if use_class_weight and source_weights is not None and target_weights is not None:
        s_cls_criterion = nn.CrossEntropyLoss(weight=source_weights)
        t_cls_criterion = nn.CrossEntropyLoss(weight=target_weights)
        cls_loss = s_cls_criterion(classifier(ccle_z), s_labels) + t_cls_criterion(classifier(tcga_z), t_labels)
    else:
        cls_criterion = nn.CrossEntropyLoss()
        cls_loss = cls_criterion(classifier(ccle_z), s_labels) + cls_criterion(classifier(tcga_z), t_labels)
    loss_log["cls_only_loss"] = cls_loss
    cls_loss.backward()
    optimizer.step()
    scheduler.step()
    classifier.eval()
    return loss_log


def train_d_ae(
    s_batch,
    t_batch,
    s_labels,
    t_labels,
    shared_encoder,
    sencoder,
    tencoder,
    discrim,
    classifier,
    optimizer,
    scheduler,
    gan_lambda_cls: float,
    num_classes: int,
    lambda_adv_eff: float = 1.0,
    lambda_proto_eff: float = 0.0,
    proto_temperature: float = 0.2,
    proto_min_samples_per_class: int = 1,
    proto_min_samples_per_domain: int = 1,
    proto_mode: str = "combined",
    proto_direction: str = "symmetric",
    proto_detach: bool = True,
    lambda_cmmd_eff: float = 0.0,
    cmmd_min_samples_per_domain: int = 2,
    cmmd_gamma="median",
    lambda_class_gap_eff: float = 0.0,
    class_gap_metric: str = "cosine",
    class_gap_min_samples_per_domain: int = 2,
    class_gap_detach_source: bool = True,
    class_gap_detach_target: bool = False,
    class_gap_l2_squared: bool = True,
    subspace_cfg=None,
    lambda_tumor_topology_eff: float = 0.0,
    tumor_topology_metric: str = "cosine_distance",
    tumor_topology_loss_type: str = "smooth_l1",
    tumor_topology_min_samples_per_domain: int = 2,
    tumor_topology_detach_source: bool = True,
    tumor_topology_normalize_distance: bool = True,
    lambda_tumor_supcon_eff: float = 0.0,
    tumor_supcon_temperature: float = 1.0,
    tumor_supcon_min_samples_per_class: int = 2,
    tumor_supcon_latent_view: str = "shared",
    lambda_tumor_var_eff: float = 0.0,
    lambda_tumor_cov_eff: float = 0.0,
    tumor_vicreg_latent_view: str = "shared",
    tumor_vicreg_var_target: float = 1.0,
    lambda_subspace_ortho_eff: float = 0.0,
    source_weights=None,
    target_weights=None,
    use_class_weight: bool = False,
):
    subspace_cfg = subspace_cfg or resolve_subspace_training_params({})
    cls_view = subspace_cfg.get("classifier_latent_view", "shared")
    topo_view = subspace_cfg.get("topology_latent_view", "shared")
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    classifier.zero_grad()
    sencoder.train()
    tencoder.train()
    shared_encoder.train()
    discrim.eval()
    classifier.train()
    optimizer.zero_grad()
    pccle_re_x, pccle_z, pccle_mu, pccle_sigma = sencoder(s_batch)
    pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, s_batch)
    ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = tencoder(t_batch)
    ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, t_batch)
    ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_encoder(s_batch)
    ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, s_batch)
    tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_encoder(t_batch)
    tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, t_batch)
    ccle_cls_z = select_latent_view(ccle_z, cls_view, subspace_cfg)
    tcga_cls_z = select_latent_view(tcga_z, cls_view, subspace_cfg)
    if use_class_weight and source_weights is not None and target_weights is not None:
        s_cls_criterion = nn.CrossEntropyLoss(weight=source_weights)
        t_cls_criterion = nn.CrossEntropyLoss(weight=target_weights)
        cls_loss = s_cls_criterion(classifier(ccle_cls_z), s_labels) + t_cls_criterion(classifier(tcga_cls_z), t_labels)
    else:
        cls_criterion = nn.CrossEntropyLoss()
        cls_loss = cls_criterion(classifier(ccle_cls_z), s_labels) + cls_criterion(classifier(tcga_cls_z), t_labels)
    pvae_loss = pccle_vae_loss + ptcga_vae_loss
    vae_loss = ccle_vae_loss + tcga_vae_loss
    o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
    g_in = alignment_discriminator_input(tcga_z, ptcga_z, subspace_cfg)
    g_loss = -torch.mean(discrim(g_in))
    proto_metrics = default_proto_metrics(mode=proto_mode, direction=proto_direction, detach=proto_detach)
    proto_loss = ccle_z.sum() * 0.0
    if lambda_proto_eff > 0:
        proto_loss, proto_metrics = compute_prototype_infonce(
            ccle_z,
            s_labels,
            tcga_z,
            t_labels,
            num_classes=num_classes,
            temperature=proto_temperature,
            min_samples_per_class=proto_min_samples_per_class,
            min_samples_per_domain=proto_min_samples_per_domain,
            mode=proto_mode,
            direction=proto_direction,
            detach_prototypes=proto_detach,
        )
    cmmd_metrics = {
        "cmmd_loss": 0.0,
        "cmmd_valid": False,
        "cmmd_valid_class_count": 0,
        "cmmd_mean_class_loss": 0.0,
    }
    cmmd_loss = ccle_z.sum() * 0.0
    if lambda_cmmd_eff > 0:
        cmmd_loss, cmmd_metrics = compute_classwise_mmd(
            ccle_z,
            s_labels,
            tcga_z,
            t_labels,
            num_classes=num_classes,
            min_samples_per_domain=cmmd_min_samples_per_domain,
            gamma=cmmd_gamma,
        )
    class_gap_metrics = {
        "class_gap_loss": 0.0,
        "class_gap_valid": False,
        "class_gap_metric": class_gap_metric,
        "class_gap_valid_class_count": 0,
        "class_gap_mean": 0.0,
        "class_gap_median": 0.0,
        "class_gap_max": 0.0,
        "class_gap_min": 0.0,
    }
    class_gap_loss = ccle_z.sum() * 0.0
    if lambda_class_gap_eff > 0:
        class_gap_loss, class_gap_metrics = compute_classwise_prototype_gap(
            ccle_z,
            s_labels,
            tcga_z,
            t_labels,
            num_classes=num_classes,
            min_samples_per_domain=class_gap_min_samples_per_domain,
            metric=class_gap_metric,
            detach_source=class_gap_detach_source,
            detach_target=class_gap_detach_target,
            l2_squared=class_gap_l2_squared,
        )
    topo_metrics = {
        "tumor_topology_loss": 0.0,
        "tumor_topology_valid": False,
        "tumor_topology_valid_class_count": 0,
    }
    tumor_topology_loss = ccle_z.sum() * 0.0
    if lambda_tumor_topology_eff > 0:
        z_src_topo = select_latent_view(ccle_z, topo_view, subspace_cfg)
        z_tgt_topo = select_latent_view(tcga_z, topo_view, subspace_cfg)
        tumor_topology_loss, topo_metrics = compute_tumor_topology_loss(
            z_src_topo,
            s_labels,
            z_tgt_topo,
            t_labels,
            num_classes=num_classes,
            min_samples_per_domain=tumor_topology_min_samples_per_domain,
            metric=tumor_topology_metric,
            topology_loss_type=tumor_topology_loss_type,
            detach_source=tumor_topology_detach_source,
            normalize_distance=tumor_topology_normalize_distance,
        )
    supcon_metrics = {"tumor_supcon_loss": 0.0, "tumor_supcon_valid": False}
    tumor_supcon_loss = ccle_z.sum() * 0.0
    if lambda_tumor_supcon_eff > 0:
        z_src_sc = select_latent_view(ccle_z, tumor_supcon_latent_view, subspace_cfg)
        z_tgt_sc = select_latent_view(tcga_z, tumor_supcon_latent_view, subspace_cfg)
        tumor_supcon_loss, supcon_metrics = compute_within_domain_supcon_loss(
            z_src_sc,
            s_labels,
            z_tgt_sc,
            t_labels,
            temperature=tumor_supcon_temperature,
            min_samples_per_class=tumor_supcon_min_samples_per_class,
        )
    vicreg_metrics = {
        "tumor_vicreg_var_loss": 0.0,
        "tumor_vicreg_cov_loss": 0.0,
        "tumor_vicreg_valid": False,
    }
    tumor_var_loss = ccle_z.sum() * 0.0
    tumor_cov_loss = ccle_z.sum() * 0.0
    if lambda_tumor_var_eff > 0 or lambda_tumor_cov_eff > 0:
        z_src_v = select_latent_view(ccle_z, tumor_vicreg_latent_view, subspace_cfg)
        z_tgt_v = select_latent_view(tcga_z, tumor_vicreg_latent_view, subspace_cfg)
        z_vicreg = torch.cat([z_src_v, z_tgt_v], dim=0)
        tumor_var_loss, tumor_cov_loss, vicreg_metrics = compute_vicreg_var_cov_loss(
            z_vicreg, var_target=tumor_vicreg_var_target
        )
    subspace_ortho_loss = ccle_z.sum() * 0.0
    if lambda_subspace_ortho_eff > 0 and subspace_cfg.get("use_tumor_subspace", False):
        z_t_s, z_tr_s = split_tumor_transfer_latent(ccle_z, subspace_cfg["tumor_dim"])
        z_t_t, z_tr_t = split_tumor_transfer_latent(tcga_z, subspace_cfg["tumor_dim"])
        subspace_ortho_loss = 0.5 * (
            compute_subspace_orthogonality_loss(z_t_s, z_tr_s)
            + compute_subspace_orthogonality_loss(z_t_t, z_tr_t)
        )
    combined_proto_cmmd = bool(lambda_proto_eff > 0 and lambda_cmmd_eff > 0)
    loss = (
        o_loss
        + float(lambda_adv_eff) * g_loss
        + vae_loss
        + pvae_loss
        + gan_lambda_cls * cls_loss
        + float(lambda_proto_eff) * proto_loss
        + float(lambda_cmmd_eff) * cmmd_loss
        + float(lambda_class_gap_eff) * class_gap_loss
        + float(lambda_tumor_topology_eff) * tumor_topology_loss
        + float(lambda_tumor_supcon_eff) * tumor_supcon_loss
        + float(lambda_tumor_var_eff) * tumor_var_loss
        + float(lambda_tumor_cov_eff) * tumor_cov_loss
        + float(lambda_subspace_ortho_eff) * subspace_ortho_loss
    )
    loss_log.update({
        "ortho_loss": o_loss,
        "pvae_loss": pvae_loss,
        "gen_loss": g_loss,
        "vae_loss": vae_loss,
        "cls_loss": cls_loss,
        "lambda_cls_eff": gan_lambda_cls,
        "lambda_adv_eff": float(lambda_adv_eff),
        "lambda_proto_eff": float(lambda_proto_eff),
        "lambda_cmmd_eff": float(lambda_cmmd_eff),
        "lambda_class_gap_eff": float(lambda_class_gap_eff),
        "class_gap_loss": class_gap_metrics.get("class_gap_loss", 0.0),
        "class_gap_metric": class_gap_metrics.get("class_gap_metric", class_gap_metric),
        "class_gap_valid": float(class_gap_metrics.get("class_gap_valid", False)),
        "class_gap_valid_class_count": class_gap_metrics.get("class_gap_valid_class_count", 0),
        "class_gap_mean": class_gap_metrics.get("class_gap_mean", 0.0),
        "class_gap_median": class_gap_metrics.get("class_gap_median", 0.0),
        "class_gap_max": class_gap_metrics.get("class_gap_max", 0.0),
        "class_gap_min": class_gap_metrics.get("class_gap_min", 0.0),
        "proto_loss": proto_metrics["proto_loss"],
        "proto_acc": proto_metrics["proto_acc"],
        "proto_valid_class_count": proto_metrics["proto_valid_class_count"],
        "proto_valid_sample_count": proto_metrics["proto_valid_sample_count"],
        "proto_mean_positive_similarity": proto_metrics["proto_mean_positive_similarity"],
        "proto_mean_negative_similarity": proto_metrics["proto_mean_negative_similarity"],
        "proto_margin": proto_metrics["proto_margin"],
        "proto_cross_domain_valid": float(proto_metrics.get("proto_cross_domain_valid", False)),
        "proto_t2s_loss": proto_metrics.get("proto_t2s_loss", 0.0),
        "proto_s2t_loss": proto_metrics.get("proto_s2t_loss", 0.0),
        "proto_t2s_acc": proto_metrics.get("proto_t2s_acc", 0.0),
        "proto_s2t_acc": proto_metrics.get("proto_s2t_acc", 0.0),
        "proto_t2s_valid_sample_count": proto_metrics.get("proto_t2s_valid_sample_count", 0),
        "proto_s2t_valid_sample_count": proto_metrics.get("proto_s2t_valid_sample_count", 0),
        "proto_source_valid_class_count": proto_metrics.get("proto_source_valid_class_count", 0),
        "proto_target_valid_class_count": proto_metrics.get("proto_target_valid_class_count", 0),
        "cmmd_loss": cmmd_metrics.get("cmmd_loss", 0.0),
        "cmmd_valid": float(cmmd_metrics.get("cmmd_valid", False)),
        "cmmd_valid_class_count": cmmd_metrics.get("cmmd_valid_class_count", 0),
        "cmmd_mean_class_loss": cmmd_metrics.get("cmmd_mean_class_loss", 0.0),
        "combined_proto_cmmd": float(combined_proto_cmmd),
        "proto_valid": float(proto_metrics["proto_valid"]),
        "lambda_tumor_topology_eff": float(lambda_tumor_topology_eff),
        "lambda_tumor_supcon_eff": float(lambda_tumor_supcon_eff),
        "lambda_tumor_var_eff": float(lambda_tumor_var_eff),
        "lambda_tumor_cov_eff": float(lambda_tumor_cov_eff),
        "lambda_subspace_ortho_eff": float(lambda_subspace_ortho_eff),
        "subspace_ortho_loss": float(subspace_ortho_loss.detach().item()),
        **{k: (float(v) if isinstance(v, (bool, int, float)) else v) for k, v in topo_metrics.items()},
        **{k: (float(v) if isinstance(v, (bool, int, float)) else v) for k, v in supcon_metrics.items()},
        **{k: (float(v) if isinstance(v, (bool, int, float)) else v) for k, v in vicreg_metrics.items()},
    })
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss_log


def _plot_pretrain_curves(train_csv, eval_csv, save_dir):
    try:
        train_df = pd.read_csv(train_csv)
        eval_df = pd.read_csv(eval_csv)
        if "epoch" in train_df.columns and CURVE_SKIP_INITIAL_EPOCHS > 0:
            train_df = train_df[train_df["epoch"] > CURVE_SKIP_INITIAL_EPOCHS]
        if "epoch" in eval_df.columns and CURVE_SKIP_INITIAL_EPOCHS > 0:
            eval_df = eval_df[eval_df["epoch"] > CURVE_SKIP_INITIAL_EPOCHS]
        if train_df.empty or eval_df.empty:
            return
        plt.figure(figsize=(10, 6))
        plot_cols = ["ortholoss", "pVAE_loss", "VAE_loss", "cls_loss", "lambda_cls_eff"]
        for col in plot_cols:
            if col in train_df.columns and col in eval_df.columns:
                plt.plot(train_df["epoch"], train_df[col], label=f"train_{col}")
                plt.plot(eval_df["epoch"], eval_df[col], "--", label=f"eval_{col}")
            elif col in train_df.columns:
                plt.plot(train_df["epoch"], train_df[col], label=f"train_{col}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.2)
        plt.title("Pretrain Learning Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pretrain_learning_curve.png"), dpi=250)
        plt.close()
    except Exception:
        return


def _plot_gan_curves(d_csv, g_csv, save_dir):
    try:
        d_df = pd.read_csv(d_csv)
        g_df = pd.read_csv(g_csv)
        if "epoch" in d_df.columns and CURVE_SKIP_INITIAL_EPOCHS > 0:
            d_df = d_df[d_df["epoch"] > CURVE_SKIP_INITIAL_EPOCHS]
        if "epoch" in g_df.columns and CURVE_SKIP_INITIAL_EPOCHS > 0:
            g_df = g_df[g_df["epoch"] > CURVE_SKIP_INITIAL_EPOCHS]
        if d_df.empty and g_df.empty:
            return
        plt.figure(figsize=(10, 5))
        for col in ["discrim_loss", "discrim_total_loss", "g_p", "d_source_score", "d_target_score"]:
            if col in d_df.columns:
                plt.plot(d_df["epoch"], d_df[col], label=col)
        for col in ["gen_loss", "cls_loss", "cls_only_loss", "lambda_cls_eff"]:
            if col in g_df.columns:
                plt.plot(g_df["epoch"], g_df[col], label=col)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.2)
        plt.title("GAN Learning Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gan_learning_curve.png"), dpi=250)
        plt.close()
    except Exception:
        return


def run_single_experiment(sourcedata, targetdata, param, exp_name, exp_dir, ccle_df_for_latent, tcga_df_for_latent):
    print(f"start experiment {exp_name}")
    use_class_weight = param.get("use_class_weight", False)
    trainloss_csv = os.path.join(exp_dir, "pretrain_loss.csv")
    evalloss_csv = os.path.join(exp_dir, "pretrain_eval_loss.csv")
    dloss_csv = os.path.join(exp_dir, "d_loss.csv")
    genloss_csv = os.path.join(exp_dir, "g_loss.csv")
    sourcetrainloader, sourcetest, source_test_labels = sourcedata[0], sourcedata[1], sourcedata[2]
    targettrainloader, targettest, target_test_labels = targetdata[0], targetdata[1], targetdata[2]
    if use_class_weight:
        source_weights, target_weights, mapping_int2str = sourcedata[3], targetdata[3], sourcedata[4]
    else:
        source_weights = None
        target_weights = None
        mapping_int2str = sourcedata[3]
    config_payload = {
        "exp_id": exp_name,
        "device": str(device),
        "params": _json_safe(param),
        "use_class_weight": use_class_weight,
    }
    with open(os.path.join(exp_dir, "params.json"), "w") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)
    num_classes = len(mapping_int2str)
    input_size = sourcetest.shape[1]
    latent_size = param.get("latent_size", 32)
    encoder_hidden_dims = param["encoder_dims"]
    decoder_hidden_dims = encoder_hidden_dims[::-1]
    dropout_rate = param["dropout_rate"]
    lambda_cls = param["lambda_cls"]
    subspace_cfg = resolve_subspace_training_params(param)
    cls_dim = classifier_input_dim(subspace_cfg)
    shared_vae = MODEL_BACKBONE(input_size=input_size, output_size=input_size, latent_size=latent_size, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, dop=dropout_rate, act_fn=nn.ReLU).to(device)
    source_private_vae = MODEL_BACKBONE(input_size=input_size, output_size=input_size, latent_size=latent_size, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, dop=dropout_rate, act_fn=nn.ReLU).to(device)
    target_private_vae = MODEL_BACKBONE(input_size=input_size, output_size=input_size, latent_size=latent_size, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, dop=dropout_rate, act_fn=nn.ReLU).to(device)
    cancer_classifier = PrimaryClassifier(input_dim=cls_dim, num_classes=num_classes, hidden_dims=[64, 32], dop=0.2, act_fn=nn.ReLU).to(device)
    shared_vae.apply(init_weights)
    source_private_vae.apply(init_weights)
    target_private_vae.apply(init_weights)
    cancer_classifier.apply(init_weights)
    source_dict = copy.deepcopy(source_private_vae.state_dict())
    shared_dict = copy.deepcopy(shared_vae.state_dict())
    target_dict = copy.deepcopy(target_private_vae.state_dict())
    classifier_dict = copy.deepcopy(cancer_classifier.state_dict())
    pretrain_epochs = param["pretrain_num_epochs"]
    pre_lr = param["pretrain_learning_rate"]
    pre_tol = 0
    pre_tol_max = param.get("pretrain_patience", 50)
    min_eval_loss = float("inf")
    models = [shared_vae, source_private_vae, target_private_vae, cancer_classifier]
    optimizer = torch.optim.Adam(chain(*(m.parameters() for m in models)), lr=pre_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, pretrain_epochs))
    if use_class_weight and source_weights is not None and target_weights is not None:
        s_cls_criterion = nn.CrossEntropyLoss(weight=source_weights)
        t_cls_criterion = nn.CrossEntropyLoss(weight=target_weights)
    else:
        cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(pretrain_epochs):
        lambda_cls_eff = get_lambda_cls_eff(epoch + 1, param)
        train_ol, train_pv, train_v, train_c, train_lce = 0.0, 0.0, 0.0, 0.0, 0.0
        steps = 0
        target_cycle = cycle(targettrainloader)
        for ccledata, ccle_labels in sourcetrainloader:
            tcgadata, tcga_labels = next(target_cycle)
            optimizer.zero_grad()
            pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccledata)
            ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcgadata)
            ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccledata)
            tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcgadata)
            p_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccledata) + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcgadata)
            vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccledata) + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcgadata)
            o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
            cls_view = subspace_cfg.get("classifier_latent_view", "shared")
            ccle_cls_z = select_latent_view(ccle_z, cls_view, subspace_cfg)
            tcga_cls_z = select_latent_view(tcga_z, cls_view, subspace_cfg)
            if use_class_weight and source_weights is not None and target_weights is not None:
                cls_loss = s_cls_criterion(cancer_classifier(ccle_cls_z), ccle_labels) + t_cls_criterion(cancer_classifier(tcga_cls_z), tcga_labels)
            else:
                cls_loss = cls_criterion(cancer_classifier(ccle_cls_z), ccle_labels) + cls_criterion(cancer_classifier(tcga_cls_z), tcga_labels)
            loss = o_loss + vae_loss + p_vae_loss + lambda_cls_eff * cls_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_ol += _to_scalar(o_loss)
            train_pv += _to_scalar(p_vae_loss)
            train_v += _to_scalar(vae_loss)
            train_c += _to_scalar(cls_loss)
            train_lce += lambda_cls_eff
            steps += 1
        append_csv_log(trainloss_csv, {
            "epoch": epoch + 1,
            "ortholoss": train_ol / max(1, steps),
            "pVAE_loss": train_pv / max(1, steps),
            "VAE_loss": train_v / max(1, steps),
            "cls_loss": train_c / max(1, steps),
            "lambda_cls_eff": train_lce / max(1, steps),
        })
        with torch.no_grad():
            pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(sourcetest)
            ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(targettest)
            ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(sourcetest)
            tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(targettest)
            eval_p = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, sourcetest) + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, targettest)
            eval_v = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, sourcetest) + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, targettest)
            eval_o = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
            ccle_cls_z = select_latent_view(ccle_z, subspace_cfg.get("classifier_latent_view", "shared"), subspace_cfg)
            tcga_cls_z = select_latent_view(tcga_z, subspace_cfg.get("classifier_latent_view", "shared"), subspace_cfg)
            if use_class_weight and source_weights is not None and target_weights is not None:
                eval_cls = s_cls_criterion(cancer_classifier(ccle_cls_z), source_test_labels) + t_cls_criterion(cancer_classifier(tcga_cls_z), target_test_labels)
            else:
                eval_cls = cls_criterion(cancer_classifier(ccle_cls_z), source_test_labels) + cls_criterion(cancer_classifier(tcga_cls_z), target_test_labels)
            eval_total = eval_o + eval_p + eval_v + lambda_cls_eff * eval_cls
            append_csv_log(evalloss_csv, {
                "epoch": epoch + 1,
                "ortholoss": _to_scalar(eval_o),
                "pVAE_loss": _to_scalar(eval_p),
                "VAE_loss": _to_scalar(eval_v),
                "cls_loss": _to_scalar(eval_cls),
                "lambda_cls_eff": lambda_cls_eff,
            })
            if _to_scalar(eval_total) < min_eval_loss:
                min_eval_loss = _to_scalar(eval_total)
                pre_tol = 0
                source_dict = copy.deepcopy(source_private_vae.state_dict())
                target_dict = copy.deepcopy(target_private_vae.state_dict())
                shared_dict = copy.deepcopy(shared_vae.state_dict())
                classifier_dict = copy.deepcopy(cancer_classifier.state_dict())
            else:
                pre_tol += 1
                if pre_tol >= pre_tol_max:
                    print(f"pretrain early stop @ epoch {epoch + 1}")
                    break
    _plot_pretrain_curves(trainloss_csv, evalloss_csv, exp_dir)
    shared_vae.load_state_dict(shared_dict)
    source_private_vae.load_state_dict(source_dict)
    target_private_vae.load_state_dict(target_dict)
    cancer_classifier.load_state_dict(classifier_dict)
    gan_epoch = param["train_num_epochs"]
    gan_lr = param["gan_learning_rate"]
    discrim = Discriminator(input_dim=discriminator_input_dim(subspace_cfg), dop=dropout_rate).to(device)
    discrim.apply(init_weights)
    gan_cfg = resolve_gan_training_params(param, gan_lr, lambda_cls)
    gan_gen_update_interval = gan_cfg["gan_gen_update_interval"]
    gan_cls_update_every_step = gan_cfg["gan_cls_update_every_step"]
    gan_cls_lr = gan_cfg["gan_cls_learning_rate"]
    gan_lambda_cls = gan_cfg["gan_lambda_cls"]
    gan_gp_weight = gan_cfg["gan_gp_weight"]
    proto_cfg = resolve_proto_training_params(param)
    class_gap_cfg = resolve_class_gap_training_params(param)
    cmmd_cfg = resolve_cmmd_training_params(param)
    topology_cfg = resolve_tumor_topology_training_params(param)
    supcon_cfg = resolve_tumor_supcon_training_params(param)
    vicreg_cfg = resolve_tumor_vicreg_training_params(param)
    lambda_adv = proto_cfg["lambda_adv"]
    proto_temperature = proto_cfg["proto_temperature"]
    proto_min_samples_per_class = proto_cfg["proto_min_samples_per_class"]
    proto_min_samples_per_domain = proto_cfg["proto_min_samples_per_domain"]
    proto_mode = proto_cfg["proto_mode"]
    proto_direction = proto_cfg["proto_direction"]
    proto_detach = proto_cfg["proto_detach"]
    cmmd_min_samples_per_domain = cmmd_cfg["cmmd_min_samples_per_domain"]
    cmmd_gamma = cmmd_cfg["cmmd_gamma"]
    best_proto_loss = float("inf")
    best_proto_margin = float("-inf")
    best_proto_acc = 0.0
    best_class_gap_loss = float("inf")
    class_gap_loss_history = []
    class_gap_valid_steps = 0
    class_gap_total_steps = 0

    discrim_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=gan_lr)
    discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discrim_optimizer, max(1, gan_epoch))
    classifier_optimizer = torch.optim.RMSprop(cancer_classifier.parameters(), lr=gan_cls_lr)
    classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, max(1, gan_epoch))
    d_ae_optimizer = torch.optim.RMSprop(
        chain(
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters(),
            cancer_classifier.parameters(),
        ),
        lr=gan_lr,
    )
    d_ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_ae_optimizer, max(1, gan_epoch))
    max_gan_tolerance = param.get("gan_patience", 20)
    gan_early_stop_metric = str(param.get("gan_early_stop_metric", "loss")).lower()
    if gan_early_stop_metric not in {"loss", "fid"}:
        raise ValueError(
            f"Unsupported gan_early_stop_metric={gan_early_stop_metric}. "
            f"Use one of: loss, fid."
        )
    gan_early_stop_min_delta = float(param.get("gan_early_stop_min_delta", 0.0))
    gan_early_stop_start_epoch = int(param.get("gan_early_stop_start_epoch", 1))
    gan_tolerance = 0
    lambda_proto_cfg = float(proto_cfg["lambda_proto"])
    post_proto_min_epoch = post_proto_checkpoint_min_epoch(param)
    gan_best_epoch_overall = 0
    gan_best_score_overall = float("inf")
    gan_best_loss_overall = float("inf")
    gan_best_epoch_post_proto = 0
    gan_best_score_post_proto = float("inf")
    gan_best_loss_post_proto = float("inf")
    shared_vae_aftergan_dict = copy.deepcopy(shared_vae.state_dict())
    classifier_aftergan_dict = copy.deepcopy(cancer_classifier.state_dict())
    source_vae_aftergan_dict = copy.deepcopy(source_private_vae.state_dict())
    target_vae_aftergan_dict = copy.deepcopy(target_private_vae.state_dict())
    discrim_aftergan_dict = copy.deepcopy(discrim.state_dict())
    shared_vae_post_proto_dict = copy.deepcopy(shared_vae.state_dict())
    classifier_post_proto_dict = copy.deepcopy(cancer_classifier.state_dict())
    source_vae_post_proto_dict = copy.deepcopy(source_private_vae.state_dict())
    target_vae_post_proto_dict = copy.deepcopy(target_private_vae.state_dict())
    discrim_post_proto_dict = copy.deepcopy(discrim.state_dict())
    for epoch in range(gan_epoch):
        gan_epoch_idx = epoch + 1
        lambda_proto_eff = get_lambda_proto_eff(gan_epoch_idx, param)
        lambda_cmmd_eff = get_lambda_cmmd_eff(gan_epoch_idx, param)
        lambda_class_gap_eff = get_lambda_class_gap_eff(gan_epoch_idx, param)
        lambda_tumor_topology_eff = get_lambda_tumor_topology_eff(gan_epoch_idx, param)
        lambda_tumor_supcon_eff = get_lambda_tumor_supcon_eff(gan_epoch_idx, param)
        lambda_tumor_var_eff = get_lambda_tumor_var_eff(gan_epoch_idx, param)
        lambda_tumor_cov_eff = get_lambda_tumor_cov_eff(gan_epoch_idx, param)
        lambda_subspace_ortho_eff = get_lambda_subspace_ortho_eff(gan_epoch_idx, param)
        dloss_list = []
        genloss_list = []
        cls_only_list = []
        target_cycle = cycle(targettrainloader)
        for step, (ccledata, ccle_labels) in enumerate(sourcetrainloader):
            tcgadata, tcga_labels = next(target_cycle)
            dloss_list.append(
                train_discrim(
                    ccledata,
                    tcgadata,
                    shared_vae,
                    source_private_vae,
                    target_private_vae,
                    discrim,
                    discrim_optimizer,
                    discrim_scheduler,
                    gan_gp_weight=gan_gp_weight,
                    subspace_cfg=subspace_cfg,
                )
            )
            if gan_cls_update_every_step:
                cls_log = train_classifier_step(
                    ccledata,
                    tcgadata,
                    ccle_labels,
                    tcga_labels,
                    shared_vae,
                    cancer_classifier,
                    classifier_optimizer,
                    classifier_scheduler,
                    source_weights=source_weights,
                    target_weights=target_weights,
                    use_class_weight=use_class_weight,
                    subspace_cfg=subspace_cfg,
                )
                cls_only_list.append(_to_scalar(cls_log.get("cls_only_loss", 0.0)))
            if (step + 1) % gan_gen_update_interval == 0:
                genloss_list.append(
                    train_d_ae(
                        ccledata,
                        tcgadata,
                        ccle_labels,
                        tcga_labels,
                        shared_vae,
                        source_private_vae,
                        target_private_vae,
                        discrim,
                        cancer_classifier,
                        d_ae_optimizer,
                        d_ae_scheduler,
                        gan_lambda_cls=gan_lambda_cls,
                        num_classes=num_classes,
                        lambda_adv_eff=lambda_adv,
                        lambda_proto_eff=lambda_proto_eff,
                        proto_temperature=proto_temperature,
                        proto_min_samples_per_class=proto_min_samples_per_class,
                        proto_min_samples_per_domain=proto_min_samples_per_domain,
                        proto_mode=proto_mode,
                        proto_direction=proto_direction,
                        proto_detach=proto_detach,
                        lambda_cmmd_eff=lambda_cmmd_eff,
                        cmmd_min_samples_per_domain=cmmd_min_samples_per_domain,
                        cmmd_gamma=cmmd_gamma,
                        lambda_class_gap_eff=lambda_class_gap_eff,
                        class_gap_metric=class_gap_cfg["class_gap_metric"],
                        class_gap_min_samples_per_domain=class_gap_cfg["class_gap_min_samples_per_domain"],
                        class_gap_detach_source=class_gap_cfg["class_gap_detach_source"],
                        class_gap_detach_target=class_gap_cfg["class_gap_detach_target"],
                        class_gap_l2_squared=class_gap_cfg["class_gap_l2_squared"],
                        subspace_cfg=subspace_cfg,
                        lambda_tumor_topology_eff=lambda_tumor_topology_eff,
                        tumor_topology_metric=topology_cfg["tumor_topology_metric"],
                        tumor_topology_loss_type=topology_cfg["tumor_topology_loss_type"],
                        tumor_topology_min_samples_per_domain=topology_cfg["tumor_topology_min_samples_per_domain"],
                        tumor_topology_detach_source=topology_cfg["tumor_topology_detach_source"],
                        tumor_topology_normalize_distance=topology_cfg["tumor_topology_normalize_distance"],
                        lambda_tumor_supcon_eff=lambda_tumor_supcon_eff,
                        tumor_supcon_temperature=supcon_cfg["tumor_supcon_temperature"],
                        tumor_supcon_min_samples_per_class=supcon_cfg["tumor_supcon_min_samples_per_class"],
                        tumor_supcon_latent_view=supcon_cfg["tumor_supcon_latent_view"],
                        lambda_tumor_var_eff=lambda_tumor_var_eff,
                        lambda_tumor_cov_eff=lambda_tumor_cov_eff,
                        tumor_vicreg_latent_view=vicreg_cfg["tumor_vicreg_latent_view"],
                        tumor_vicreg_var_target=vicreg_cfg["tumor_vicreg_var_target"],
                        lambda_subspace_ortho_eff=lambda_subspace_ortho_eff,
                        source_weights=source_weights,
                        target_weights=target_weights,
                        use_class_weight=use_class_weight,
                    )
                )
        if not dloss_list:
            continue
        dloss_mean = _mean_loss_logs(dloss_list)
        genloss_mean = _mean_loss_logs(genloss_list)
        if cls_only_list:
            genloss_mean["cls_only_loss"] = float(np.mean(cls_only_list))
        genloss_mean["lambda_cls_eff"] = gan_lambda_cls
        genloss_mean["lambda_adv_eff"] = lambda_adv
        genloss_mean["lambda_proto_eff"] = lambda_proto_eff
        genloss_mean["lambda_cmmd_eff"] = lambda_cmmd_eff
        genloss_mean["lambda_class_gap"] = class_gap_cfg["lambda_class_gap"]
        genloss_mean["lambda_class_gap_eff"] = lambda_class_gap_eff
        genloss_mean["gan_gen_update_interval"] = gan_gen_update_interval
        if lambda_class_gap_eff > 0:
            class_gap_total_steps += 1
            if genloss_mean.get("class_gap_valid"):
                class_gap_valid_steps += 1
                cg_loss = float(genloss_mean.get("class_gap_loss", float("inf")))
                class_gap_loss_history.append(cg_loss)
                if cg_loss < best_class_gap_loss:
                    best_class_gap_loss = cg_loss
        genloss_mean["proto_mode"] = proto_mode
        genloss_mean["proto_direction"] = proto_direction
        genloss_mean["proto_detach"] = float(proto_detach)
        dloss_mean["epoch"] = epoch + 1
        genloss_mean["epoch"] = epoch + 1
        if genloss_mean.get("proto_valid"):
            proto_loss_epoch = float(genloss_mean.get("proto_loss", float("inf")))
            proto_margin_epoch = float(genloss_mean.get("proto_margin", float("-inf")))
            proto_acc_epoch = float(genloss_mean.get("proto_acc", 0.0))
            if proto_loss_epoch < best_proto_loss:
                best_proto_loss = proto_loss_epoch
            if proto_margin_epoch > best_proto_margin:
                best_proto_margin = proto_margin_epoch
            if proto_acc_epoch > best_proto_acc:
                best_proto_acc = proto_acc_epoch
        append_csv_log(dloss_csv, dloss_mean)
        temp_loss = sum(
            v for k, v in dloss_mean.items() if k != "epoch" and isinstance(v, (int, float))
        ) + sum(
            v for k, v in genloss_mean.items() if k != "epoch" and isinstance(v, (int, float))
        )
        if gan_early_stop_metric == "fid":
            with torch.no_grad():
                _, source_epoch_z, _, _ = shared_vae(sourcetest)
                _, target_epoch_z, _, _ = shared_vae(targettest)
            source_epoch_latent = source_epoch_z.detach().cpu().numpy()
            target_epoch_latent = target_epoch_z.detach().cpu().numpy()
            current_score = _compute_fid(source_epoch_latent, target_epoch_latent)
        else:
            current_score = temp_loss
        genloss_mean["early_stop_metric"] = gan_early_stop_metric
        genloss_mean["early_stop_score"] = current_score
        genloss_mean["temp_loss"] = temp_loss
        append_csv_log(genloss_csv, genloss_mean)
        monitor_active = (epoch + 1) >= gan_early_stop_start_epoch
        if current_score < (gan_best_score_overall - gan_early_stop_min_delta):
            gan_best_score_overall = current_score
            gan_best_loss_overall = temp_loss
            gan_tolerance = 0
            gan_best_epoch_overall = epoch + 1
            shared_vae_aftergan_dict = copy.deepcopy(shared_vae.state_dict())
            classifier_aftergan_dict = copy.deepcopy(cancer_classifier.state_dict())
            source_vae_aftergan_dict = copy.deepcopy(source_private_vae.state_dict())
            target_vae_aftergan_dict = copy.deepcopy(target_private_vae.state_dict())
            discrim_aftergan_dict = copy.deepcopy(discrim.state_dict())
        elif monitor_active:
            gan_tolerance += 1
            if gan_tolerance >= max_gan_tolerance:
                print(f"gan early stop @ epoch {epoch + 1}")
                break
        post_proto_eligible = (
            lambda_proto_cfg > 0
            and lambda_proto_eff > 0
            and gan_epoch_idx >= post_proto_min_epoch
        )
        if post_proto_eligible and current_score < (gan_best_score_post_proto - gan_early_stop_min_delta):
            gan_best_score_post_proto = current_score
            gan_best_loss_post_proto = temp_loss
            gan_best_epoch_post_proto = gan_epoch_idx
            shared_vae_post_proto_dict = copy.deepcopy(shared_vae.state_dict())
            classifier_post_proto_dict = copy.deepcopy(cancer_classifier.state_dict())
            source_vae_post_proto_dict = copy.deepcopy(source_private_vae.state_dict())
            target_vae_post_proto_dict = copy.deepcopy(target_private_vae.state_dict())
            discrim_post_proto_dict = copy.deepcopy(discrim.state_dict())
    _plot_gan_curves(dloss_csv, genloss_csv, exp_dir)
    use_post_proto = lambda_proto_cfg > 0 and gan_best_epoch_post_proto >= post_proto_min_epoch
    if use_post_proto:
        sel_shared = shared_vae_post_proto_dict
        sel_classifier = classifier_post_proto_dict
        sel_source = source_vae_post_proto_dict
        sel_target = target_vae_post_proto_dict
        sel_discrim = discrim_post_proto_dict
        best_gan_epoch = gan_best_epoch_post_proto
        best_gan_loss = gan_best_loss_post_proto
        best_gan_score = gan_best_score_post_proto
    else:
        sel_shared = shared_vae_aftergan_dict
        sel_classifier = classifier_aftergan_dict
        sel_source = source_vae_aftergan_dict
        sel_target = target_vae_aftergan_dict
        sel_discrim = discrim_aftergan_dict
        best_gan_epoch = gan_best_epoch_overall
        best_gan_loss = gan_best_loss_overall
        best_gan_score = gan_best_score_overall
    shared_vae.load_state_dict(sel_shared)
    source_private_vae.load_state_dict(sel_source)
    target_private_vae.load_state_dict(sel_target)
    discrim.load_state_dict(sel_discrim)
    cancer_classifier.load_state_dict(sel_classifier)
    torch.save(sel_shared, os.path.join(exp_dir, "after_traingan_shared_vae.pth"))
    torch.save(sel_source, os.path.join(exp_dir, "after_traingan_source_vae.pth"))
    torch.save(sel_target, os.path.join(exp_dir, "after_traingan_target_vae.pth"))
    torch.save(sel_classifier, os.path.join(exp_dir, "after_traingan_classifier.pth"))
    torch.save(sel_discrim, os.path.join(exp_dir, "after_traingan_discriminator.pth"))
    torch.save(shared_vae_aftergan_dict, os.path.join(exp_dir, "after_traingan_overall_shared_vae.pth"))
    torch.save(classifier_aftergan_dict, os.path.join(exp_dir, "after_traingan_overall_classifier.pth"))
    ccle_latent_dict = _encode_latent_dict(shared_vae, ccle_df_for_latent)
    tcga_latent_raw_dict = _encode_latent_dict(shared_vae, tcga_df_for_latent)
    tcga_latent_dict = tcga_latent_raw_dict
    tcga_latent_dict = deduplicate_tcga_latent_dict(tcga_latent_dict)
    with open(os.path.join(exp_dir, "ccle_latent_dict.pkl"), "wb") as f:
        pickle.dump(ccle_latent_dict, f)
    with open(os.path.join(exp_dir, "tcga_latent_dict.pkl"), "wb") as f:
        pickle.dump(tcga_latent_dict, f)
    source_latent = np.asarray(list(ccle_latent_dict.values()), dtype=np.float32)
    target_latent = np.asarray(list(tcga_latent_dict.values()), dtype=np.float32)
    with torch.no_grad():
        _, source_test_z, _, _ = shared_vae(sourcetest)
        _, target_test_z, _, _ = shared_vae(targettest)
    source_true = source_test_labels.detach().cpu().numpy()
    target_true = target_test_labels.detach().cpu().numpy()
    source_test_latent = source_test_z.detach().cpu().numpy()
    target_test_latent = target_test_z.detach().cpu().numpy()
    cluster_metrics = _kmeans_combined_metrics(
        source_test_latent,
        target_test_latent,
        source_true,
        target_true,
        len(mapping_int2str),
    )
    proto_guard = compute_proto_checkpoint_guard(
        param,
        gan_best_epoch_overall,
        gan_best_epoch_post_proto,
        gan_best_loss_overall,
        gan_best_loss_post_proto if gan_best_epoch_post_proto > 0 else None,
    )
    structure_metrics = compute_proto_structure_metrics(
        source_test_latent,
        source_true,
        target_test_latent,
        target_true,
        len(mapping_int2str),
        float(cluster_metrics.get("kmeans_ari", 0.0)),
        float(cluster_metrics.get("kmeans_silhouette", 0.0)),
        min_samples_per_domain=proto_min_samples_per_domain,
    )
    metrics = {
        "exp_id": exp_name,
        "best_gan_epoch": best_gan_epoch,
        "best_gan_loss": best_gan_loss,
        "best_proto_loss": None if best_proto_loss == float("inf") else best_proto_loss,
        "best_proto_margin": None if best_proto_margin == float("-inf") else best_proto_margin,
        "best_proto_acc": best_proto_acc if best_proto_acc > 0 else None,
        "proto_mode": proto_mode,
        "proto_direction": proto_direction,
        "proto_detach": proto_detach,
        "proto_pair_align": proto_cfg.get("proto_pair_align", False),
        "lambda_cmmd": cmmd_cfg["lambda_cmmd"],
        "combined_proto_cmmd": bool(float(param.get("lambda_proto", 0)) > 0 and cmmd_cfg["lambda_cmmd"] > 0),
        "lambda_class_gap": class_gap_cfg["lambda_class_gap"],
        "class_gap_metric": class_gap_cfg["class_gap_metric"],
        "class_gap_start_epoch": class_gap_cfg["class_gap_start_epoch"],
        "class_gap_full_epoch": class_gap_cfg["class_gap_full_epoch"],
        "best_class_gap_loss": None if best_class_gap_loss == float("inf") else best_class_gap_loss,
        "mean_class_gap_loss": float(np.mean(class_gap_loss_history)) if class_gap_loss_history else None,
        "class_gap_valid_rate": (
            float(class_gap_valid_steps) / float(class_gap_total_steps) if class_gap_total_steps else None
        ),
        "lambda_tumor_topology": topology_cfg["lambda_tumor_topology"],
        "tumor_topology_metric": topology_cfg["tumor_topology_metric"],
        "lambda_tumor_supcon": supcon_cfg["lambda_tumor_supcon"],
        "lambda_tumor_var": vicreg_cfg["lambda_tumor_var"],
        "lambda_tumor_cov": vicreg_cfg["lambda_tumor_cov"],
        "use_tumor_subspace": subspace_cfg["use_tumor_subspace"],
        "tumor_dim": subspace_cfg["tumor_dim"],
        "lambda_subspace_ortho": subspace_cfg["lambda_subspace_ortho"],
        **proto_guard,
        **structure_metrics,
        "gan_early_stop_metric": gan_early_stop_metric,
        "gan_early_stop_best_score": best_gan_score,
        "gan_early_stop_min_delta": gan_early_stop_min_delta,
        "gan_early_stop_start_epoch": gan_early_stop_start_epoch,
        "gan_patience": max_gan_tolerance,
        "fid": _compute_fid(source_latent),
        "mmd": _calculate_mmd(source_latent, target_latent),
        "wasserstein": _calculate_wasserstein(source_latent, target_latent),
        "tcga_raw_sample_count_for_latent": int(len(tcga_latent_raw_dict)),
        "tcga_patient_count_for_latent": int(len(tcga_latent_dict)),
    }
    metrics.update(cluster_metrics)
    with open(os.path.join(exp_dir, "gan_metrics.json"), "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "gan_metrics.csv"), index=False)
    _plot_gan_tsne(
        source_test_z.detach().cpu().numpy(),
        target_test_z.detach().cpu().numpy(),
        source_true,
        target_true,
        mapping_int2str,
        os.path.join(exp_dir, "tsne_gan_best.png"),
    )
    with open(os.path.join(exp_dir, "run_summary.json"), "w") as f:
        json.dump(_json_safe({
            "exp_id": exp_name,
            "params": param,
            "metrics": metrics,
            "artifacts": {
                "weights": [
                    "after_traingan_shared_vae.pth",
                    "after_traingan_source_vae.pth",
                    "after_traingan_target_vae.pth",
                    "after_traingan_classifier.pth",
                    "after_traingan_discriminator.pth"
                ],
                "latents": ["ccle_latent_dict.pkl", "tcga_latent_dict.pkl"],
                "plots": ["tsne_gan_best.png", "gan_learning_curve.png", "pretrain_learning_curve.png"],
            },
        }), f, indent=2)
    return metrics


def load_params_grid(config_path="config/params_grid.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    if "pretrain_param_combinations" in config:
        return config["pretrain_param_combinations"], "combinations"
    return config.get("pretrain_params", {}), "grid"


def _load_full_feature_frames(ccle_path, tcga_path):
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    tcga_df = pd.read_csv(tcga_path, index_col=0)
    ccle_df.index = ccle_df.index.astype(str)
    tcga_df.index = tcga_df.index.astype(str)
    # Keep full-frame export aligned with training feature space.
    common_cols = [c for c in ccle_df.columns if c in set(tcga_df.columns)]
    if len(common_cols) == 0:
        raise ValueError(
            f"No overlapping feature columns between source ({ccle_path}) "
            f"and target ({tcga_path}) for latent export."
        )
    ccle_df = ccle_df.loc[:, common_cols]
    tcga_df = tcga_df.loc[:, common_cols]
    return ccle_df, tcga_df


def _append_model_select(outfolder, row: Dict):
    model_select_path = os.path.join(outfolder, PRETRAIN_MODEL_SELECT_FILENAME)
    df_new = pd.DataFrame([row])
    if os.path.exists(model_select_path) and os.path.getsize(model_select_path) > 0:
        try:
            df_old = pd.read_csv(model_select_path)
        except pd.errors.EmptyDataError:
            df_old = pd.DataFrame()
        if df_old.empty:
            df = df_new
        else:
            df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(model_select_path, index=False)


def main():
    parser = argparse.ArgumentParser("pretrain_VAEwC")
    parser.add_argument("--outfolder", default="./result/pretrain", type=str, help="output folder")
    parser.add_argument("--target_domain", default="tcga", choices=["tcga", "pdx"], type=str, help="target domain selection")
    parser.add_argument("--target", default=None, type=str, help="target expression csv path (optional, auto by target_domain if not set)")
    parser.add_argument("--target_response", default=None, type=str, help="target response csv path (optional, auto by target_domain if not set)")
    parser.add_argument("--target_cancer_reference", default=None, type=str, help="target cancer reference csv path (optional, auto by target_domain if not set)")
    parser.add_argument("--config", default="config/params_grid.json", type=str, help="path to params grid")
    parser.add_argument(
        "--overlap_tcga",
        default=None,
        type=str,
        help="overlap patient list to exclude from TCGA training (only used when target_domain=tcga)",
    )
    parser.add_argument(
        "--ccle-info",
        default=DEFAULT_CCLE_INFO_CSV,
        type=str,
        help="CCLE sample info CSV with cancer_type column",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override epochs to tiny values for pipeline smoke testing",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Override training batch size (default: from config or 128)",
    )
    args = parser.parse_args()
    domain_cfg = TARGET_DOMAIN_CONFIG[args.target_domain]
    resolved_target = args.target or domain_cfg["target_expression"]
    resolved_target_response = args.target_response or domain_cfg["target_response"]
    resolved_target_cancer_ref = args.target_cancer_reference or domain_cfg["target_cancer_reference"]
    source_path = DEFAULT_SOURCE_CSV
    if args.target_domain == "tcga":
        overlap_path = args.overlap_tcga
    else:
        overlap_path = None
    params_payload, payload_type = load_params_grid(args.config)
    if payload_type == "combinations":
        param_list = params_payload
    else:
        keys, values = zip(*params_payload.items())
        param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    safemakedirs(args.outfolder)
    all_rows = []
    tmp_suffix = f"{os.getpid()}_{int(time.time() * 1000)}"
    training_target_path, removed_count = _prepare_training_target_csv(
        resolved_target, overlap_path, args.outfolder, tmp_suffix=tmp_suffix
    )
    tmp_training_path = os.path.join(args.outfolder, f"_tmp_target_for_training_{tmp_suffix}.csv")
    try:
        if overlap_path:
            if removed_count > 0:
                print(f"[TCGA overlap filter] removed {removed_count} rows for training target")
            else:
                print("[TCGA overlap filter] overlap file provided but no rows removed; use original target data")
        else:
            print("[TCGA overlap filter] disabled (no --overlap_tcga provided), use original target data")
        frame_cache = {}
        for param_dict in param_list:
            param_dict = dict(param_dict)
            if args.smoke_test:
                param_dict["pretrain_num_epochs"] = 1
                param_dict["train_num_epochs"] = 2
                param_dict["gan_patience"] = 1
                param_dict["pretrain_patience"] = 1
            if args.batch_size is not None and args.batch_size > 0:
                param_dict["batch_size"] = int(args.batch_size)
            cache_key = f"{source_path}|{resolved_target}"
            if cache_key not in frame_cache:
                frame_cache[cache_key] = _load_full_feature_frames(source_path, resolved_target)
            ccle_df_full, tcga_df_full = frame_cache[cache_key]
            effective_batch = _cap_batch_size(
                param_dict.get("batch_size", 128),
                min(len(ccle_df_full), len(tcga_df_full)),
            )
            param_dict["batch_size"] = effective_batch
            sourcedata, targetdata = _load_labeled_data_patient_aware(
                ccle_path=source_path,
                xena_path=training_target_path,
                batch_size=effective_batch,
                use_class_weight=param_dict.get("use_class_weight", False),
                target_domain=args.target_domain,
                target_cancer_reference_path=resolved_target_cancer_ref,
                ccle_info_path=args.ccle_info,
            )
            exp_name, exp_dir = _next_experiment_dir(args.outfolder)
            metrics = run_single_experiment(
                sourcedata=sourcedata,
                targetdata=targetdata,
                param=param_dict,
                exp_name=exp_name,
                exp_dir=exp_dir,
                ccle_df_for_latent=ccle_df_full,
                tcga_df_for_latent=tcga_df_full,
            )
            row = build_experiment_summary_row(param_dict, exp_name, metrics)
            _append_model_select(args.outfolder, row)
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(os.path.join(args.outfolder, "summary_results.csv"), index=False)
        print(
            f"All experiments done. {PRETRAIN_MODEL_SELECT_FILENAME} and "
            f"summary_results.csv saved under {args.outfolder}"
        )
    finally:
        if training_target_path == tmp_training_path and os.path.exists(tmp_training_path):
            os.remove(tmp_training_path)
            print("[cleanup] removed temporary training target csv")


if __name__ == "__main__":
    main()
