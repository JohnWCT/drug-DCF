"""
Temporary archive for functions removed from active scripts.

Keep this file as a safety stash during refactoring; functions can be restored
or re-homed later if needed.
"""

import os
import json
import numpy as np
import pandas as pd
from tools.data_utils import normalize_gene_expression


def build_filtered_domain_frames_legacy(ccle_path, xena_path, normalization_method="zscore"):
    """
    Legacy helper moved from pretrain_VAEwC.py (unused in current pipeline).
    """
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    xena_df = pd.read_csv(xena_path, index_col=0)
    ccle_df = normalize_gene_expression(ccle_df, method=normalization_method)
    xena_df = normalize_gene_expression(xena_df, method=normalization_method)
    ccle_df.index = ccle_df.index.astype(str)
    xena_df.index = xena_df.index.astype(str)
    ccle_info_path = os.path.join("data", "ccle_sample_info_df.csv")
    xena_info_path = os.path.join("data", "xena_sample_info_df.csv")
    if not os.path.exists(ccle_info_path) or not os.path.exists(xena_info_path):
        src_y = np.zeros(len(ccle_df), dtype=np.int64)
        tgt_y = np.zeros(len(xena_df), dtype=np.int64)
        return ccle_df, xena_df, src_y, tgt_y
    ccle_info = pd.read_csv(ccle_info_path, index_col=0, header=0)
    xena_info = pd.read_csv(xena_info_path, index_col=0, header=0)
    ccle_info.index = ccle_info.index.astype(str)
    xena_info.index = xena_info.index.astype(str)
    target_to_source_map = {
        "acute myeloid leukemia": "na",
        "adrenocortical cancer": "na",
        "bladder urothelial carcinoma": "Bladder Cancer",
        "brain lower grade glioma": "Brain Cancer",
        "breast invasive carcinoma": "Breast Cancer",
        "cervical & endocervical cancer": "Cervical Cancer",
        "cholangiocarcinoma": "Bile Duct Cancer",
        "colon adenocarcinoma": "Colon/Colorectal Cancer",
        "diffuse large B-cell lymphoma": "na",
        "esophageal carcinoma": "Esophageal Cancer",
        "glioblastoma multiforme": "Brain Cancer",
        "head & neck squamous cell carcinoma": "Head and Neck Cancer",
        "kidney chromophobe": "Kidney Cancer",
        "kidney clear cell carcinoma": "Kidney Cancer",
        "kidney papillary cell carcinoma": "Kidney Cancer",
        "liver hepatocellular carcinoma": "Liver Cancer",
        "lung adenocarcinoma": "Lung Cancer",
        "lung squamous cell carcinoma": "Lung Cancer",
        "mesothelioma": "na",
        "ovarian serous cystadenocarcinoma": "Ovarian Cancer",
        "pancreatic adenocarcinoma": "Pancreatic Cancer",
        "pheochromocytoma & paraganglioma": "na",
        "prostate adenocarcinoma": "Prostate Cancer",
        "rectum adenocarcinoma": "Colon/Colorectal Cancer",
        "sarcoma": "Sarcoma",
        "skin cutaneous melanoma": "Skin Cancer",
        "stomach adenocarcinoma": "Gastric Cancer",
        "testicular germ cell tumor": "na",
        "thymoma": "na",
        "thyroid carcinoma": "Thyroid Cancer",
        "uterine carcinosarcoma": "Endometrial/Uterine Cancer",
        "uterine corpus endometrioid carcinoma": "Endometrial/Uterine Cancer",
        "uveal melanoma": "Eye Cancer",
    }
    xena_info["_primary_disease"] = xena_info["_primary_disease"].map(target_to_source_map)
    excluded = list(set(ccle_df.index) - set(ccle_info.index))
    low_count = ccle_info.primary_disease.value_counts()[ccle_info.primary_disease.value_counts() < 10].index
    excluded.extend(list(ccle_info[ccle_info.primary_disease.isin(low_count)].index))
    ccle_ok = ccle_df.loc[ccle_df.index.difference(excluded)]
    src_idx = ccle_ok.index.intersection(ccle_info[ccle_info.primary_disease.notna()].index)
    ccle_ok = ccle_ok.loc[src_idx]
    src_labels = ccle_info.loc[src_idx, "primary_disease"]
    tgt_idx = xena_df.index.intersection(xena_info[xena_info["_primary_disease"].notna()].index)
    xena_ok = xena_df.loc[tgt_idx]
    tgt_labels = xena_info.loc[tgt_idx, "_primary_disease"]
    common = set(src_labels.unique()) & set(tgt_labels.unique())
    common = {label for label in common if label != "na" and not pd.isna(label)}
    ccle_ok = ccle_ok[src_labels.isin(common)]
    src_labels = src_labels[src_labels.isin(common)]
    xena_ok = xena_ok[tgt_labels.isin(common)]
    tgt_labels = tgt_labels[tgt_labels.isin(common)]
    label_map = {d: i for i, d in enumerate(sorted(list(common)))}
    src_y = np.array([label_map[v] for v in src_labels])
    tgt_y = np.array([label_map[v] for v in tgt_labels])
    return ccle_ok, xena_ok, src_y, tgt_y


def load_config_all_split_legacy(
    config_path="config/params_grid.json", include_lambda_cls=False
):
    """
    Legacy helper moved from step1_finetune_pipeline_All_split.py (unused in current CLI flow).
    """
    default_params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        "pretrain_learning_rate": [0.001],
        "gan_learning_rate": [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
        "dropout_rate": [0.0, 0.2],
        "encoder_dims": [[512, 256, 128, 64], [128]],
        "lambda_cls": [1, 5, 10],
    }
    default_finetune_grid = {"ftlr": [0.01, 0.001], "scheduler_flag": [True, False]}
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if not include_lambda_cls and "lambda_cls" in config["pretrain_params"]:
            del config["pretrain_params"]["lambda_cls"]
        return config
    except Exception:
        defaults = {
            "pretrain_params": default_params_grid.copy(),
            "finetune_params": default_finetune_grid,
        }
        if not include_lambda_cls and "lambda_cls" in defaults["pretrain_params"]:
            del defaults["pretrain_params"]["lambda_cls"]
        return defaults
