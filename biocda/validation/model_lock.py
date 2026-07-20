"""Final model lock manifest."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from biocda.utils.hashing import sha256_file, sha256_json


def build_model_lock_manifest(
    *,
    outcome_status: str,
    model_name: Optional[str],
    architecture_version: str,
    split_seeds: List[int],
    checkpoint_paths: List[str],
    git_commit: str,
    config_hash: str,
    dataset_hash: str,
    split_manifest_hash: str,
    encoder_hashes: Dict[str, str],
) -> Dict[str, Any]:
    checkpoint_hashes = []
    for p in checkpoint_paths:
        path = Path(p)
        checkpoint_hashes.append(sha256_file(path) if path.is_file() else "missing")

    return {
        "status": outcome_status,
        "model_name": model_name,
        "architecture_version": architecture_version,
        "selection_protocol": "repeated_unseen_drug",
        "selection_metric": "drug_macro_auc",
        "checkpoint_policy": "best_per_split_seed",
        "split_seeds": split_seeds,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_hashes": checkpoint_hashes,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "split_manifest_hash": split_manifest_hash,
        "omics_encoder_hash": encoder_hashes.get("omics", ""),
        "drug_encoder_hash": encoder_hashes.get("drug", ""),
        "context_artifact_hash": encoder_hashes.get("context", ""),
        "tcga_used_for_selection": False,
        "attention_interface": {
            "per_head_probabilities": True,
            "pre_softmax_logits": True,
            "atom_mask": True,
            "model_atom_index": True,
            "original_atom_index": True,
            "rdkit_atom_index": True,
            "atom_ptr": True,
        },
    }


def write_model_lock_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
