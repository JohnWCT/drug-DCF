"""Shared utilities for Round 9 deconfounding QC and conditional diagnostics."""

from __future__ import annotations

import json
import os
import pickle
import re
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tools.optimization_selection import DEFAULT_FORCE_BASELINE_PATHS
from tools.pretrain_common import TARGET_DOMAIN_CONFIG, tcga_three_segment_key

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CCLE_INFO_CSV = os.path.join("data", "ccle_sample_info_df.csv")

KNOWN_BASELINE_HINTS = {
    **DEFAULT_FORCE_BASELINE_PATHS,
    "exp_188": "result/optimization_runs/vaewc_round8A_control_arch_broad/pretrain/exp_188",
}

CHECKPOINT_MARKERS = (
    "gan_metrics.json",
    "after_traingan_shared_vae.pth",
    "run_summary.json",
)
LATENT_PKL_NAMES = ("ccle_latent_dict.pkl", "tcga_latent_dict.pkl")
LATENT_CSV_PATTERNS = (
    "gan_latent_best.csv",
    "latent_best.csv",
    "latents/*.csv",
    "embeddings/*.csv",
)
TSNE_PATTERNS = ("tsne_gan_best.png", "tsne_*.png")

RESOLVED_BASELINE_COLUMNS = [
    "exp_id",
    "role",
    "required",
    "resolved",
    "checkpoint_dir",
    "config_path",
    "params_path",
    "summary_path",
    "latent_path",
    "source_run_dir",
    "source_round",
    "latent_size",
    "encoder_dims",
    "dropout_rate",
    "lambda_cls",
    "lambda_tumor_var",
    "lambda_tumor_cov",
    "lambda_proto",
    "lambda_class_gap",
    "lambda_tumor_topology",
    "lambda_tumor_supcon",
    "notes",
]


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def relpath_from_root(path: str) -> str:
    return os.path.relpath(resolve_path(path), PROJECT_ROOT)


def load_json(path: str) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_completeness_score(checkpoint_dir: str) -> int:
    score = 0
    for marker in CHECKPOINT_MARKERS:
        if os.path.exists(os.path.join(checkpoint_dir, marker)):
            score += 1
    for name in LATENT_PKL_NAMES:
        if os.path.exists(os.path.join(checkpoint_dir, name)):
            score += 2
    return score


def _read_params_from_checkpoint(checkpoint_dir: str) -> dict:
    params_path = os.path.join(checkpoint_dir, "params.json")
    if not os.path.exists(params_path):
        return {}
    payload = load_json(params_path)
    return payload.get("params", payload)


def _infer_source_round(checkpoint_dir: str) -> str:
    rel = relpath_from_root(checkpoint_dir)
    m = re.search(r"optimization_runs/([^/]+)", rel)
    return m.group(1) if m else ""


def build_resolved_row(
    exp_id: str,
    role: str,
    required: bool,
    checkpoint_dir: str,
    notes: str = "",
) -> dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    params = _read_params_from_checkpoint(checkpoint_dir)
    latent_path = ""
    for name in LATENT_PKL_NAMES:
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            latent_path = relpath_from_root(p)
            break
    return {
        "exp_id": exp_id,
        "role": role,
        "required": required,
        "resolved": True,
        "checkpoint_dir": relpath_from_root(checkpoint_dir),
        "config_path": "",
        "params_path": relpath_from_root(os.path.join(checkpoint_dir, "params.json"))
        if os.path.exists(os.path.join(checkpoint_dir, "params.json"))
        else "",
        "summary_path": relpath_from_root(os.path.join(checkpoint_dir, "run_summary.json"))
        if os.path.exists(os.path.join(checkpoint_dir, "run_summary.json"))
        else "",
        "latent_path": latent_path,
        "source_run_dir": relpath_from_root(os.path.dirname(os.path.dirname(checkpoint_dir)))
        if "pretrain" in checkpoint_dir
        else relpath_from_root(os.path.dirname(checkpoint_dir)),
        "source_round": _infer_source_round(checkpoint_dir),
        "latent_size": params.get("latent_size", ""),
        "encoder_dims": str(params.get("encoder_dims", "")),
        "dropout_rate": params.get("dropout_rate", ""),
        "lambda_cls": params.get("lambda_cls", ""),
        "lambda_tumor_var": params.get("lambda_tumor_var", ""),
        "lambda_tumor_cov": params.get("lambda_tumor_cov", ""),
        "lambda_proto": params.get("lambda_proto", ""),
        "lambda_class_gap": params.get("lambda_class_gap", ""),
        "lambda_tumor_topology": params.get("lambda_tumor_topology", ""),
        "lambda_tumor_supcon": params.get("lambda_tumor_supcon", ""),
        "notes": notes,
    }


def load_exp_metrics(checkpoint_dir: str) -> dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    metrics: dict = {}
    for fname in ("gan_metrics.json", "run_summary.json"):
        path = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(path):
            continue
        payload = load_json(path)
        if fname == "run_summary.json":
            metrics.update(payload.get("metrics", {}))
        else:
            metrics.update(payload)
    return metrics


def find_existing_tsne_path(checkpoint_dir: str) -> str:
    checkpoint_dir = resolve_path(checkpoint_dir)
    for pattern in TSNE_PATTERNS:
        matches = sorted(glob(os.path.join(checkpoint_dir, pattern)))
        if matches:
            return relpath_from_root(matches[0])
    return ""


def _load_latent_dict(path: str) -> Dict[str, np.ndarray]:
    with open(resolve_path(path), "rb") as f:
        data = pickle.load(f)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in data.items()}


def find_latent_paths(checkpoint_dir: str) -> Tuple[Optional[str], Optional[str]]:
    checkpoint_dir = resolve_path(checkpoint_dir)
    source_path = os.path.join(checkpoint_dir, "ccle_latent_dict.pkl")
    target_path = os.path.join(checkpoint_dir, "tcga_latent_dict.pkl")
    if os.path.exists(source_path) and os.path.exists(target_path):
        return source_path, target_path
    for pattern in LATENT_CSV_PATTERNS:
        for path in glob(os.path.join(checkpoint_dir, pattern)):
            if "ccle" in os.path.basename(path).lower():
                source_path = path
            if "tcga" in os.path.basename(path).lower():
                target_path = path
    return (
        source_path if source_path and os.path.exists(source_path) else None,
        target_path if target_path and os.path.exists(target_path) else None,
    )


def _resolve_latent_vector(latent_dict: Dict[str, np.ndarray], sample_id: str) -> Optional[np.ndarray]:
    sid = str(sample_id)
    if sid in latent_dict:
        return latent_dict[sid]
    patient_key = tcga_three_segment_key(sid)
    if patient_key in latent_dict:
        return latent_dict[patient_key]
    return None


def _load_cancer_maps() -> Tuple[pd.Series, pd.Series]:
    ccle_info = pd.read_csv(resolve_path(DEFAULT_CCLE_INFO_CSV), index_col=0)
    ccle_info.index = ccle_info.index.astype(str)
    ccle_map = ccle_info["cancer_type"].astype(str).str.strip()

    tcga_ref = resolve_path(TARGET_DOMAIN_CONFIG["tcga"]["target_cancer_reference"])
    xena_info = pd.read_csv(tcga_ref, index_col=0)
    xena_info.index = xena_info.index.astype(str)
    cancer_col = "cancer_type" if "cancer_type" in xena_info.columns else "cancerType"
    tcga_map = xena_info[cancer_col].astype(str).str.strip()
    return ccle_map, tcga_map


def load_latent_domain_frame(checkpoint_dir: str) -> pd.DataFrame:
    """Return long-form latent table: sample_id, domain, cancer_type, z columns."""
    source_pkl, target_pkl = find_latent_paths(checkpoint_dir)
    if not source_pkl or not target_pkl:
        raise FileNotFoundError(f"Missing latent pickles under {checkpoint_dir}")

    source_latent = _load_latent_dict(source_pkl)
    target_latent = _load_latent_dict(target_pkl)
    ccle_map, tcga_map = _load_cancer_maps()

    rows: List[dict] = []
    for sid, vec in source_latent.items():
        if sid not in ccle_map.index:
            continue
        row = {"sample_id": sid, "domain": "source", "cancer_type": str(ccle_map.loc[sid])}
        for i, val in enumerate(vec):
            row[f"z{i}"] = float(val)
        rows.append(row)
    for sid, vec in target_latent.items():
        key = sid
        if key not in tcga_map.index:
            key = tcga_three_segment_key(sid)
        if key not in tcga_map.index:
            continue
        row = {"sample_id": sid, "domain": "target", "cancer_type": str(tcga_map.loc[key])}
        for i, val in enumerate(vec):
            row[f"z{i}"] = float(val)
        rows.append(row)
    if not rows:
        raise ValueError(f"No labeled latent rows for {checkpoint_dir}")
    return pd.DataFrame(rows)


def latent_matrix_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_cols = [c for c in df.columns if c.startswith("z")]
    x = df[z_cols].to_numpy(dtype=np.float32)
    domain = (df["domain"] == "target").astype(int).to_numpy()
    cancer = df["cancer_type"].astype(str).to_numpy()
    return x, domain, cancer


def leakage_strength(auc: float) -> float:
    if auc is None or (isinstance(auc, float) and np.isnan(auc)):
        return float("nan")
    return float(abs(float(auc) - 0.5))


def fit_domain_classifier(
    x: np.ndarray,
    y: np.ndarray,
    classifier: str,
    random_state: int = 42,
) -> Tuple[float, float]:
    if len(np.unique(y)) < 2 or len(x) < 4:
        return float("nan"), float("nan")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=random_state, stratify=y
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=random_state
        )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if classifier == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == "small_mlp":
        model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    else:
        pred = model.predict(x_test)
        auc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, model.predict(x_test)))
    return auc, bal_acc


def fit_cancer_classifier(x: np.ndarray, cancer_labels: np.ndarray, random_state: int = 42) -> Tuple[float, float]:
    if len(np.unique(cancer_labels)) < 2 or len(x) < 4:
        return float("nan"), float("nan")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, cancer_labels, test_size=0.25, random_state=random_state, stratify=cancer_labels
        )
    except ValueError:
        return float("nan"), float("nan")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    return macro_f1, bal_acc


def macro_mean(values: Sequence[float]) -> float:
    arr = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        num += float(v) * float(w)
        den += float(w)
    return num / den if den > 0 else float("nan")


def classify_deconfounding_qc(
    global_auc: float,
    macro_cond_auc: float,
    cancer_macro_f1: float,
    inter_cancer_margin: float,
) -> str:
    global_good = not np.isnan(global_auc) and leakage_strength(global_auc) < 0.15
    cond_low = not np.isnan(macro_cond_auc) and leakage_strength(macro_cond_auc) < 0.12
    cancer_good = not np.isnan(cancer_macro_f1) and cancer_macro_f1 >= 0.35
    margin_good = not np.isnan(inter_cancer_margin) and inter_cancer_margin > 0.1
    if np.isnan(global_auc) and np.isnan(macro_cond_auc):
        return "insufficient_evidence"
    if global_good and cond_low and cancer_good and margin_good:
        return "good_conditional_deconfounding"
    if global_good and not cond_low:
        return "global_only_alignment"
    if global_good and (not cancer_good or not margin_good):
        return "biology_collapse_risk"
    return "insufficient_evidence"


def effective_rank(x: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    x_centered = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x_centered, compute_uv=False)
    s = s[s > 1e-8]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def iter_reproduction_models(run_dir: str, manifest_path: Optional[str] = None) -> List[dict]:
    run_dir = resolve_path(run_dir)
    manifest_path = manifest_path or os.path.join(run_dir, "manifests", "pretrain_sweep_manifest.csv")
    if not os.path.exists(manifest_path):
        return []
    df = pd.read_csv(manifest_path)
    models = []
    for _, row in df[df["status"] == "success"].iterrows():
        result_dir = row.get("result_dir", "")
        if not isinstance(result_dir, str) or not result_dir.strip():
            continue
        exp_dir = resolve_path(result_dir)
        model_id = os.path.basename(exp_dir.rstrip(os.sep))
        models.append(
            {
                "job_id": row["job_id"],
                "model_id": model_id,
                "checkpoint_dir": exp_dir,
                "source_exp_id": row.get("source_exp_id", ""),
                "role": row.get("source_role", ""),
                "reproduction_seed": row.get("reproduction_seed", ""),
            }
        )
    return models


def write_csv(df: pd.DataFrame, path: str) -> str:
    path = resolve_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_md(path: str, lines: Iterable[str]) -> str:
    path = resolve_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
