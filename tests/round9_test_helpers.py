"""Shared fixtures for Round 9 tests."""
from __future__ import annotations
import json
import os
from typing import Dict

def write_minimal_checkpoint(root: str, exp_id: str, params: Dict | None = None) -> str:
    exp_dir = os.path.join(root, "pretrain", exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    params = params or {
        "latent_size": 4,
        "encoder_dims": [8, 4],
        "dropout_rate": 0.1,
        "lambda_cls": 20,
        "pretrain_num_epochs": 2,
        "train_num_epochs": 2,
        "random_seed": 101,
    }
    with open(os.path.join(exp_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump({"exp_id": exp_id, "params": params}, f)
    metrics = {"fid": 1.0, "wasserstein": 0.5, "kmeans_ari": 0.3}
    with open(os.path.join(exp_dir, "gan_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    with open(os.path.join(exp_dir, "after_traingan_shared_vae.pth"), "w") as f:
        f.write("stub")
    return exp_dir

def make_round9_baseline_config(path: str) -> None:
    payload = {
        "seeds": [101, 202, 303],
        "baselines": [
            {"exp_id": "exp_048", "role": "primary", "required": True, "explicit_path": None},
            {"exp_id": "exp_missing", "role": "optional", "required": False, "explicit_path": None},
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
