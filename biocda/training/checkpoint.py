"""Checkpoint save/load with strict loading and legacy D0 conversion."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class CheckpointLoadingReport:
    loaded_keys: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    ignored_legacy_keys: List[str] = field(default_factory=list)
    architecture_version: Optional[str] = None
    model_type: Optional[str] = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_biocda_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: Dict[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_type: str = "cross_attention",
    architecture_version: str = "biocda-xa-v1",
    omics_encoder_sha: str = "",
    drug_encoder_sha: str = "",
    context_artifact_sha: str = "",
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "config": config,
        "model_type": model_type,
        "architecture_version": architecture_version,
        "omics_encoder_sha": omics_encoder_sha,
        "drug_encoder_sha": drug_encoder_sha,
        "context_artifact_sha": context_artifact_sha,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


LEGACY_POOLING_PREFIXES = (
    "fc1_xd.",
    "out.",
    "pool.",
)


def convert_legacy_d0_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Strip legacy graph pooling / projection keys from D0 checkpoints."""
    converted: Dict[str, torch.Tensor] = {}
    ignored: List[str] = []
    for key, value in state_dict.items():
        normalized = key
        if key.startswith("encoder."):
            normalized = key[len("encoder.") :]
        if key.startswith("drug_encoder.gin."):
            normalized = key[len("drug_encoder.gin.") :]
        if any(normalized.startswith(p) for p in LEGACY_POOLING_PREFIXES):
            ignored.append(key)
            continue
        if normalized.startswith("convs.") or normalized.startswith("bns."):
            converted[f"drug_encoder.gin.{normalized}"] = value
        elif normalized.startswith("gin."):
            converted[f"drug_encoder.{normalized}"] = value
        else:
            converted[key] = value
    return converted, ignored


def load_biocda_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool = True,
    convert_legacy_d0: bool = False,
) -> CheckpointLoadingReport:
    blob = torch.load(checkpoint_path, map_location="cpu")
    state = blob.get("model_state_dict", blob)
    ignored: List[str] = []
    if convert_legacy_d0:
        state, ignored = convert_legacy_d0_checkpoint(state)
    report = model.load_state_dict(state, strict=strict)
    loading = CheckpointLoadingReport(
        loaded_keys=sorted(state.keys()),
        missing_keys=list(report.missing_keys),
        unexpected_keys=list(report.unexpected_keys),
        ignored_legacy_keys=ignored,
        architecture_version=blob.get("architecture_version"),
        model_type=blob.get("model_type"),
    )
    return loading


def write_loading_report(report: CheckpointLoadingReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
