"""
Evaluate raw expression features (no model training).

Metrics align with pretrain_VAEwC gan_metrics.csv convention:
  fid, mmd, wasserstein, tcga_raw_sample_count_for_latent,
  tcga_patient_count_for_latent, kmeans_*.

Outputs (written directly under --outfolder by default):
  raw_metrics.csv / raw_metrics.json
  tsne_raw_data.png  (A: source vs target, B: cancer type)

Use --exp_name only when you need a separate subfolder (e.g. exp_011).

Docker example:
docker exec -w /workspace/DAPL DAPL python3 evaluate_raw_data.py \
    --outfolder result/raw_eval \
    --target_domain tcga
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from tools.dataprocess import safemakedirs
from tools.pretrain_tsne import plot_latent_tsne_dual
from tools.pretrain_common import (
    TARGET_DOMAIN_CONFIG,
    deduplicate_tcga_latent_dict,
    json_safe as _json_safe,
    prepare_training_target_csv as _prepare_training_target_csv,
    tcga_three_segment_key,
)

plt.switch_backend("Agg")

DEFAULT_SOURCE_CSV = "data/pretrain_ccle.csv"


# ---------------------------------------------------------------------------
# Distribution metrics (same logic as pretrain_VAEwC.py)
# ---------------------------------------------------------------------------

def _calculate_mmd(source_feat: np.ndarray, target_feat: np.ndarray, gamma=None) -> float:
    if source_feat.shape[0] > 1000:
        source_feat = source_feat[np.random.choice(source_feat.shape[0], 1000, replace=False)]
    if target_feat.shape[0] > 1000:
        target_feat = target_feat[np.random.choice(target_feat.shape[0], 1000, replace=False)]
    if gamma is None:
        gamma = 1.0 / source_feat.shape[1]
    xx = np.exp(-gamma * cdist(source_feat, source_feat, "sqeuclidean"))
    yy = np.exp(-gamma * cdist(target_feat, target_feat, "sqeuclidean"))
    xy = np.exp(-gamma * cdist(source_feat, target_feat, "sqeuclidean"))
    return float(max(0.0, xx.mean() + yy.mean() - 2 * xy.mean()))


def _calculate_wasserstein(source_feat: np.ndarray, target_feat: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(source_feat, axis=0) - np.mean(target_feat, axis=0)))


def _sanitize_for_fid(feat: np.ndarray, name: str = "features") -> np.ndarray:
    if feat is None:
        return None
    arr = np.asarray(feat, dtype=np.float64)
    if arr.ndim != 2:
        return arr
    finite_mask = np.isfinite(arr).all(axis=1)
    dropped = int(arr.shape[0] - finite_mask.sum())
    if dropped > 0:
        print(f"[fid] warning: dropped {dropped}/{arr.shape[0]} rows with NaN/Inf from {name}")
    return arr[finite_mask]


def _compute_fid(source_feat: np.ndarray, target_feat: np.ndarray = None) -> float:
    try:
        source_clean = _sanitize_for_fid(source_feat, name="source")
        if source_clean is None or source_clean.shape[0] < 2:
            print("[fid] warning: source has <2 finite rows; returning inf")
            return float("inf")

        mu = np.mean(source_clean, axis=0)
        sigma = np.cov(source_clean, rowvar=False)

        if target_feat is None:
            prior = np.random.randn(*source_clean.shape)
            mu2 = np.mean(prior, axis=0)
            sigma2 = np.cov(prior, rowvar=False)
        else:
            target_clean = _sanitize_for_fid(target_feat, name="target")
            if target_clean is None or target_clean.shape[0] < 2:
                print("[fid] warning: target has <2 finite rows; returning inf")
                return float("inf")
            mu2 = np.mean(target_clean, axis=0)
            sigma2 = np.cov(target_clean, rowvar=False)

        if not (
            np.isfinite(mu).all()
            and np.isfinite(sigma).all()
            and np.isfinite(mu2).all()
            and np.isfinite(sigma2).all()
        ):
            print("[fid] warning: non-finite mu/sigma; returning inf")
            return float("inf")

        from tools.metrics import calculate_frechet_distance

        value = float(calculate_frechet_distance(mu, sigma, mu2, sigma2))
        if not np.isfinite(value):
            return float("inf")
        return value
    except Exception as err:
        print(f"[fid] warning: FID failed ({err}); returning inf")
        return float("inf")


def _kmeans_combined_metrics(
    source_feat: np.ndarray,
    target_feat: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    metrics = {
        "kmeans_k": np.nan,
        "kmeans_ari": np.nan,
        "kmeans_nmi": np.nan,
        "kmeans_silhouette": np.nan,
        "kmeans_calinski_harabasz": np.nan,
        "kmeans_davies_bouldin": np.nan,
    }
    if source_feat is None or target_feat is None:
        return metrics
    if len(source_feat) < 2 or len(target_feat) < 2:
        return metrics

    feat = np.vstack([source_feat, target_feat])
    labels = np.concatenate([
        np.asarray(source_labels, dtype=np.int64),
        np.asarray(target_labels, dtype=np.int64),
    ])
    if len(feat) < 3:
        return metrics

    k = int(max(2, min(n_clusters, len(np.unique(labels)), len(feat) - 1)))
    if k < 2:
        return metrics

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(feat)
    metrics["kmeans_k"] = int(k)
    metrics["kmeans_ari"] = float(adjusted_rand_score(labels, cluster_labels))
    metrics["kmeans_nmi"] = float(max(0.0, min(1.0, normalized_mutual_info_score(labels, cluster_labels))))
    try:
        metrics["kmeans_silhouette"] = float(silhouette_score(feat, cluster_labels))
    except Exception:
        metrics["kmeans_silhouette"] = np.nan
    try:
        metrics["kmeans_calinski_harabasz"] = float(calinski_harabasz_score(feat, cluster_labels))
    except Exception:
        metrics["kmeans_calinski_harabasz"] = np.nan
    try:
        metrics["kmeans_davies_bouldin"] = float(davies_bouldin_score(feat, cluster_labels))
    except Exception:
        metrics["kmeans_davies_bouldin"] = np.nan
    return metrics


# ---------------------------------------------------------------------------
# Data loading (patient-aware, mirrors pretrain_VAEwC._load_labeled_data_patient_aware)
# ---------------------------------------------------------------------------

def _norm_name(v: str) -> str:
    return str(v).strip().lower().replace("&", "and")


def load_labeled_feature_splits(
    ccle_path: str,
    xena_path: str,
    target_domain: str = "tcga",
    target_cancer_reference_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, Dict[int, str], int]:
    """
    Returns source_test, target_test, test labels (int), mapping_int2str, num_classes.
    Uses the same filtering / label alignment as pretrain_VAEwC training.
    """
    study_to_source_map = {
        "LAML": "na", "ACC": "na", "BLCA": "Bladder Cancer", "LGG": "Brain Cancer",
        "BRCA": "Breast Cancer", "CESC": "Cervical Cancer", "CHOL": "Bile Duct Cancer",
        "LCML": "na", "COAD": "Colon/Colorectal Cancer", "CNTL": "na",
        "ESCA": "Esophageal Cancer", "FPPP": "na", "GBM": "Brain Cancer",
        "HNSC": "Head and Neck Cancer", "KICH": "Kidney Cancer", "KIRC": "Kidney Cancer",
        "KIRP": "Kidney Cancer", "LIHC": "Liver Cancer", "LUAD": "Lung Cancer",
        "LUSC": "Lung Cancer", "DLBC": "na", "MESO": "na", "MISC": "na",
        "OV": "Ovarian Cancer", "PAAD": "Pancreatic Cancer", "PCPG": "na",
        "PRAD": "Prostate Cancer", "READ": "Colon/Colorectal Cancer", "SARC": "Sarcoma",
        "SKCM": "Skin Cancer", "STAD": "Gastric Cancer", "TGCT": "na", "THYM": "na",
        "THCA": "Thyroid Cancer", "UCS": "Endometrial/Uterine Cancer",
        "UCEC": "Endometrial/Uterine Cancer", "UVM": "Eye Cancer",
    }

    ccle_df = pd.read_csv(ccle_path, index_col=0)
    xena_df = pd.read_csv(xena_path, index_col=0)
    ccle_df.index = ccle_df.index.astype(str)
    xena_df.index = xena_df.index.astype(str)
    common_cols = [c for c in ccle_df.columns if c in set(xena_df.columns)]
    if len(common_cols) == 0:
        raise ValueError(f"No overlapping features between {ccle_path} and {xena_path}")
    ccle_df = ccle_df.loc[:, common_cols]
    xena_df = xena_df.loc[:, common_cols]

    ccle_info = pd.read_csv(os.path.join("data", "ccle_sample_info_df.csv"), index_col=0, header=0)
    target_domain = str(target_domain).lower()

    if target_domain == "tcga":
        if not target_cancer_reference_path:
            target_cancer_reference_path = TARGET_DOMAIN_CONFIG["tcga"]["target_cancer_reference"]
        xena_info = pd.read_csv(target_cancer_reference_path, index_col=0, header=0)
        xena_info.index = xena_info.index.astype(str)
        study_name_to_abbr = {
            "Acute Myeloid Leukemia": "LAML", "Adrenocortical carcinoma": "ACC",
            "Bladder Urothelial Carcinoma": "BLCA", "Brain Lower Grade Glioma": "LGG",
            "Breast invasive carcinoma": "BRCA",
            "Cervical squamous cell carcinoma and endocervical adenocarcinoma": "CESC",
            "Cholangiocarcinoma": "CHOL", "Chronic Myelogenous Leukemia": "LCML",
            "Colon adenocarcinoma": "COAD", "Controls": "CNTL", "Esophageal carcinoma": "ESCA",
            "FFPE Pilot Phase II": "FPPP", "Glioblastoma multiforme": "GBM",
            "Head and Neck squamous cell carcinoma": "HNSC", "Kidney Chromophobe": "KICH",
            "Kidney renal clear cell carcinoma": "KIRC",
            "Kidney renal papillary cell carcinoma": "KIRP",
            "Liver hepatocellular carcinoma": "LIHC", "Lung adenocarcinoma": "LUAD",
            "Lung squamous cell carcinoma": "LUSC",
            "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma": "DLBC",
            "Mesothelioma": "MESO", "Miscellaneous": "MISC",
            "Ovarian serous cystadenocarcinoma": "OV", "Pancreatic adenocarcinoma": "PAAD",
            "Pheochromocytoma and Paraganglioma": "PCPG", "Prostate adenocarcinoma": "PRAD",
            "Rectum adenocarcinoma": "READ", "Sarcoma": "SARC", "Skin Cutaneous Melanoma": "SKCM",
            "Stomach adenocarcinoma": "STAD", "Testicular Germ Cell Tumors": "TGCT",
            "Thymoma": "THYM", "Thyroid carcinoma": "THCA", "Uterine Carcinosarcoma": "UCS",
            "Uterine Corpus Endometrial Carcinoma": "UCEC", "Uveal Melanoma": "UVM",
        }
        name_to_source_map = {
            _norm_name(name): study_to_source_map[abbr]
            for name, abbr in study_name_to_abbr.items()
            if abbr in study_to_source_map
        }
        target_to_source_map = {
            "acute myeloid leukemia": "na", "adrenocortical cancer": "na",
            "bladder urothelial carcinoma": "Bladder Cancer",
            "brain lower grade glioma": "Brain Cancer",
            "breast invasive carcinoma": "Breast Cancer",
            "cervical & endocervical cancer": "Cervical Cancer",
            "cholangiocarcinoma": "Bile Duct Cancer",
            "colon adenocarcinoma": "Colon/Colorectal Cancer",
            "diffuse large b-cell lymphoma": "na", "esophageal carcinoma": "Esophageal Cancer",
            "glioblastoma multiforme": "Brain Cancer",
            "head & neck squamous cell carcinoma": "Head and Neck Cancer",
            "kidney chromophobe": "Kidney Cancer",
            "kidney clear cell carcinoma": "Kidney Cancer",
            "kidney papillary cell carcinoma": "Kidney Cancer",
            "liver hepatocellular carcinoma": "Liver Cancer",
            "lung adenocarcinoma": "Lung Cancer", "lung squamous cell carcinoma": "Lung Cancer",
            "mesothelioma": "na", "ovarian serous cystadenocarcinoma": "Ovarian Cancer",
            "pancreatic adenocarcinoma": "Pancreatic Cancer",
            "pheochromocytoma & paraganglioma": "na",
            "prostate adenocarcinoma": "Prostate Cancer",
            "rectum adenocarcinoma": "Colon/Colorectal Cancer", "sarcoma": "Sarcoma",
            "skin cutaneous melanoma": "Skin Cancer", "stomach adenocarcinoma": "Gastric Cancer",
            "testicular germ cell tumor": "na", "thymoma": "na", "thyroid carcinoma": "Thyroid Cancer",
            "uterine carcinosarcoma": "Endometrial/Uterine Cancer",
            "uterine corpus endometrioid carcinoma": "Endometrial/Uterine Cancer",
            "uveal melanoma": "Eye Cancer",
        }
        name_to_source_map.update({_norm_name(k): v for k, v in target_to_source_map.items()})
        name_to_source_map.update({
            "pheochromocytoma and paraganglioma": "na",
            "head and neck squamous cell carcinoma": "Head and Neck Cancer",
            "cervical and endocervical cancer": "Cervical Cancer",
            "adrenocortical carcinoma": "na",
            "diffuse large b-cell lymphoma": "na",
            "kidney renal clear cell carcinoma": "Kidney Cancer",
            "kidney renal papillary cell carcinoma": "Kidney Cancer",
            "testicular germ cell tumors": "na",
            "uterine corpus endometrial carcinoma": "Endometrial/Uterine Cancer",
        })

        if "Study_Abbreviation" in xena_info.columns:
            label_series = xena_info["Study_Abbreviation"].astype(str).map(study_to_source_map)
        else:
            disease_col = "_primary_disease" if "_primary_disease" in xena_info.columns else None
            if disease_col is None:
                candidate_cols = [c for c in xena_info.columns if "study" in c.lower() or "disease" in c.lower()]
                disease_col = candidate_cols[0] if candidate_cols else xena_info.columns[0]
            label_series = xena_info[disease_col].astype(str).map(
                lambda x: name_to_source_map.get(_norm_name(x))
            )

        xena_info["_primary_disease"] = label_series
        xena_info["patient_id"] = xena_info.index.map(tcga_three_segment_key)
        xena_patient_info = xena_info.dropna(subset=["_primary_disease"]).sort_index().groupby("patient_id").first()
    elif target_domain == "pdx":
        if not target_cancer_reference_path:
            target_cancer_reference_path = TARGET_DOMAIN_CONFIG["pdx"]["target_cancer_reference"]
        xena_info = pd.read_csv(target_cancer_reference_path)
        pdx_to_source_map = {
            "Breast Cancer": "Breast Cancer",
            "Skin Cancer": "Skin Cancer",
            "Colon/Colorectal Cancer": "Colon/Colorectal Cancer",
            "Lung Cancer": "Lung Cancer",
            "Pancreatic Cancer": "Pancreatic Cancer",
        }
        if "Model" in xena_info.columns and "cancerType" in xena_info.columns:
            xena_info["Model"] = xena_info["Model"].astype(str)
            xena_info["_primary_disease"] = xena_info["cancerType"].astype(str).str.strip().map(pdx_to_source_map)
            xena_patient_info = (
                xena_info.dropna(subset=["_primary_disease"])
                .sort_values("Model")
                .drop_duplicates(subset=["Model"], keep="first")
                .set_index("Model")
            )
        else:
            sample_col = "Sample_id" if "Sample_id" in xena_info.columns else xena_info.columns[0]
            xena_info[sample_col] = xena_info[sample_col].astype(str)
            xena_patient_info = (
                xena_info[[sample_col]]
                .drop_duplicates(subset=[sample_col], keep="first")
                .set_index(sample_col)
            )
            xena_patient_info["_primary_disease"] = "Breast Cancer"
    else:
        raise ValueError(f"Unsupported target_domain={target_domain}")

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

    xena_df = xena_df.copy()
    xena_df["patient_id"] = xena_df.index.map(tcga_three_segment_key)
    xena_df = xena_df.sort_index().groupby("patient_id", as_index=True).first()

    valid_tcga_ids = xena_df.index.intersection(xena_patient_info.index)
    xena_df = xena_df.loc[valid_tcga_ids]
    xena_labels = xena_patient_info.loc[valid_tcga_ids, "_primary_disease"]
    common_labels = sorted(list((set(ccle_info.primary_disease.unique()) & set(xena_labels.unique())) - {"na"}))

    ccle_mask = ccle_info.primary_disease.isin(common_labels)
    xena_mask = xena_labels.isin(common_labels)
    ccle_df = ccle_df.loc[ccle_mask]
    ccle_labels = ccle_info.loc[ccle_mask, "primary_disease"]
    xena_df = xena_df.loc[xena_mask]
    xena_labels = xena_labels.loc[xena_mask]

    if len(ccle_df) == 0 or len(xena_df) == 0:
        raise ValueError("No valid labeled samples after filtering.")

    _, ccle_test, _, ccle_test_y = train_test_split(
        ccle_df, ccle_labels, test_size=test_size, stratify=ccle_labels, random_state=random_state
    )
    _, xena_test, _, xena_test_y = train_test_split(
        xena_df, xena_labels, test_size=test_size, stratify=xena_labels, random_state=random_state
    )

    label_map = {d: i for i, d in enumerate(common_labels)}
    mapping_int2str = {i: d for d, i in label_map.items()}
    ccle_test_label_int = np.array([label_map[x] for x in ccle_test_y], dtype=np.int64)
    xena_test_label_int = np.array([label_map[x] for x in xena_test_y], dtype=np.int64)

    return (
        ccle_test,
        xena_test,
        ccle_test_label_int,
        xena_test_label_int,
        mapping_int2str,
        len(common_labels),
    )


def load_full_feature_frames(ccle_path: str, tcga_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    tcga_df = pd.read_csv(tcga_path, index_col=0)
    ccle_df.index = ccle_df.index.astype(str)
    tcga_df.index = tcga_df.index.astype(str)
    common_cols = [c for c in ccle_df.columns if c in set(tcga_df.columns)]
    if len(common_cols) == 0:
        raise ValueError(f"No overlapping features between {ccle_path} and {tcga_path}")
    return ccle_df.loc[:, common_cols], tcga_df.loc[:, common_cols]


def build_tcga_patient_features(tcga_df: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
    """
    Mirror latent export counting in pretrain_VAEwC:
      - raw sample count: rows in full TCGA csv
      - patient count: one row per TCGA-XX-XXXX after dedup (same as deduplicate_tcga_latent_dict)
    """
    raw_count = int(len(tcga_df))
    raw_dict = {str(idx): tcga_df.loc[idx].values.tolist() for idx in tcga_df.index}
    patient_dict = deduplicate_tcga_latent_dict(raw_dict)
    patient_feat = np.asarray(list(patient_dict.values()), dtype=np.float32)
    return patient_feat, raw_count, int(len(patient_dict))


def plot_raw_tsne_dual(
    source_feat: np.ndarray,
    target_feat: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    mapping_int2str: Dict[int, str],
    save_path: str,
    max_points: int = 3000,
):
    if len(source_feat) == 0 or len(target_feat) == 0:
        print("[tsne] skip: empty source or target")
        return
    plot_latent_tsne_dual(
        source_feat,
        target_feat,
        source_labels,
        target_labels,
        mapping_int2str,
        save_path,
        suptitle="Raw Expression t-SNE (Test Split)",
        max_points=max_points,
    )
    print(f"[tsne] saved {save_path}")


def _resolve_output_dir(outfolder: str, exp_name: str | None) -> Tuple[str, str]:
    """
    Default: write/overwrite directly under outfolder (one run -> one result set).
    With --exp_name: write under outfolder/<exp_name>/.
    """
    safemakedirs(outfolder)
    if exp_name:
        exp_dir = os.path.join(outfolder, exp_name)
        safemakedirs(exp_dir)
        return exp_name, exp_dir
    run_id = os.path.basename(os.path.normpath(outfolder)) or "raw_eval"
    return run_id, outfolder


def run_raw_evaluation(
    source_path: str,
    target_path: str,
    target_domain: str,
    target_cancer_reference: str,
    exp_dir: str,
    exp_name: str,
) -> Dict:
    ccle_full, tcga_full = load_full_feature_frames(source_path, target_path)
    source_full = np.asarray(ccle_full.values, dtype=np.float32)
    target_patient_feat, tcga_raw_count, tcga_patient_count = build_tcga_patient_features(tcga_full)

    (
        source_test,
        target_test,
        source_test_labels,
        target_test_labels,
        mapping_int2str,
        num_classes,
    ) = load_labeled_feature_splits(
        ccle_path=source_path,
        xena_path=target_path,
        target_domain=target_domain,
        target_cancer_reference_path=target_cancer_reference,
    )

    source_test_feat = np.asarray(source_test.values, dtype=np.float32)
    target_test_feat = np.asarray(target_test.values, dtype=np.float32)

    cluster_metrics = _kmeans_combined_metrics(
        source_test_feat,
        target_test_feat,
        source_test_labels,
        target_test_labels,
        num_classes,
    )

    metrics = {
        "exp_id": exp_name,
        "evaluation_mode": "raw_expression",
        "source_csv": source_path,
        "target_csv": target_path,
        "target_domain": target_domain,
        "fid": _compute_fid(source_full),
        "mmd": _calculate_mmd(source_full, target_patient_feat),
        "wasserstein": _calculate_wasserstein(source_full, target_patient_feat),
        "tcga_raw_sample_count_for_latent": tcga_raw_count,
        "tcga_patient_count_for_latent": tcga_patient_count,
        "source_sample_count": int(len(ccle_full)),
        "source_test_count": int(len(source_test)),
        "target_test_count": int(len(target_test)),
        "feature_dim": int(source_full.shape[1]),
    }
    metrics.update(cluster_metrics)

    with open(os.path.join(exp_dir, "raw_metrics.json"), "w") as f:
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "raw_metrics.csv"), index=False)

    plot_raw_tsne_dual(
        source_test_feat,
        target_test_feat,
        source_test_labels,
        target_test_labels,
        mapping_int2str,
        os.path.join(exp_dir, "tsne_raw_data.png"),
    )

    with open(os.path.join(exp_dir, "run_summary.json"), "w") as f:
        json.dump(
            _json_safe({
                "exp_id": exp_name,
                "metrics": metrics,
                "artifacts": ["raw_metrics.csv", "raw_metrics.json", "tsne_raw_data.png"],
            }),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metrics


def main():
    parser = argparse.ArgumentParser("evaluate_raw_data")
    parser.add_argument("--outfolder", default="result/raw_eval", type=str, help="output root folder")
    parser.add_argument("--target_domain", default="tcga", choices=["tcga", "pdx"], type=str)
    parser.add_argument("--source", default=DEFAULT_SOURCE_CSV, type=str, help="source expression csv")
    parser.add_argument("--target", default=None, type=str, help="target expression csv (auto if omitted)")
    parser.add_argument("--target_cancer_reference", default=None, type=str)
    parser.add_argument(
        "--overlap_tcga",
        default=None,
        type=str,
        help="overlap patient list to exclude from TCGA (tcga only)",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="optional subfolder under --outfolder (e.g. exp_011); default writes into --outfolder directly",
    )
    args = parser.parse_args()

    domain_cfg = TARGET_DOMAIN_CONFIG[args.target_domain]
    resolved_target = args.target or domain_cfg["target_expression"]
    resolved_target_cancer_ref = args.target_cancer_reference or domain_cfg["target_cancer_reference"]

    safemakedirs(args.outfolder)
    overlap_path = args.overlap_tcga if args.target_domain == "tcga" else None
    training_target_path, removed_count = _prepare_training_target_csv(
        resolved_target, overlap_path, args.outfolder
    )
    tmp_training_path = os.path.join(args.outfolder, "_tmp_target_for_eval.csv")

    try:
        if overlap_path:
            if removed_count > 0:
                print(f"[TCGA overlap filter] removed {removed_count} rows")
            else:
                print("[TCGA overlap filter] overlap file provided but no rows removed")
        else:
            print("[TCGA overlap filter] disabled")

        exp_name, exp_dir = _resolve_output_dir(args.outfolder, args.exp_name)
        if args.exp_name:
            print(f"[output] writing to subfolder: {exp_dir}")
        else:
            print(f"[output] writing directly to (overwrite): {exp_dir}")

        metrics = run_raw_evaluation(
            source_path=args.source,
            target_path=training_target_path,
            target_domain=args.target_domain,
            target_cancer_reference=resolved_target_cancer_ref,
            exp_dir=exp_dir,
            exp_name=exp_name,
        )

        print("Raw evaluation done.")
        for k in [
            "fid", "mmd", "wasserstein",
            "tcga_raw_sample_count_for_latent", "tcga_patient_count_for_latent",
            "kmeans_k", "kmeans_ari", "kmeans_nmi",
            "kmeans_silhouette", "kmeans_calinski_harabasz", "kmeans_davies_bouldin",
        ]:
            print(f"  {k}: {metrics.get(k)}")
        print(f"Results saved under: {exp_dir}")
    finally:
        if training_target_path == tmp_training_path and os.path.exists(tmp_training_path):
            os.remove(tmp_training_path)
            print("[cleanup] removed temporary target csv")


if __name__ == "__main__":
    main()
