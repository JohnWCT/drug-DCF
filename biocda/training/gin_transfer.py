"""Transfer Round20 BioCDA-Predictive GIN (convs+bns) into XA GINAtomEncoder."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from biocda.utils.hashing import sha256_file

TRANSFER_PREFIXES = ("convs.", "bns.")
IGNORE_PREFIXES = ("fc1_xd.", "out.")
IGNORE_EXACT: Set[str] = set()


@dataclass
class GinTransferReport:
    source_checkpoint: str
    source_hash: str
    target_hash: str
    loaded_keys: List[str] = field(default_factory=list)
    ignored_keys: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    strict: bool = True
    ok: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tensor_sha(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _extract_encoder_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    msd = checkpoint.get("model_state_dict", checkpoint)
    if "encoder" in msd and hasattr(msd["encoder"], "keys"):
        return {k: v for k, v in msd["encoder"].items()}
    # Flat keys prefixed with encoder.
    flat = {}
    for k, v in msd.items():
        if k.startswith("encoder."):
            flat[k[len("encoder.") :]] = v
    if flat:
        return flat
    raise KeyError("Could not find encoder state_dict in predictive checkpoint")


def transfer_e3_gin_to_xa(
    predictive_checkpoint: Path | str,
    xa_model: nn.Module,
    *,
    strict: bool = True,
) -> GinTransferReport:
    """
    Load GINConv + BatchNorm weights from BioCDA-Predictive into XA drug_encoder.gin.

    Explicitly ignores pooling projection (fc1_xd, out) and fusion/head modules.
    Never uses silent strict=False without an audited report.
    """
    path = Path(predictive_checkpoint)
    ckpt = torch.load(path, map_location="cpu")
    source_encoder = _extract_encoder_state(ckpt)

    gin = xa_model.drug_encoder.gin
    target_state = gin.state_dict()

    loaded: Dict[str, torch.Tensor] = {}
    ignored: List[str] = []
    unexpected: List[str] = []

    for key, tensor in source_encoder.items():
        if key.startswith(IGNORE_PREFIXES) or key in IGNORE_EXACT:
            ignored.append(key)
            continue
        if not key.startswith(TRANSFER_PREFIXES):
            ignored.append(key)
            continue
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(tensor.shape) != tuple(target_state[key].shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: source={tuple(tensor.shape)} "
                f"target={tuple(target_state[key].shape)}"
            )
        loaded[key] = tensor

    required = [k for k in target_state if k.startswith(TRANSFER_PREFIXES)]
    missing = [k for k in required if k not in loaded]

    report = GinTransferReport(
        source_checkpoint=str(path),
        source_hash=sha256_file(path),
        target_hash="",
        loaded_keys=sorted(loaded.keys()),
        ignored_keys=sorted(ignored),
        missing_keys=sorted(missing),
        unexpected_keys=sorted(unexpected),
        strict=strict,
    )

    if strict and (missing or unexpected):
        report.ok = False
        report.notes.append(
            f"strict transfer failed: missing={len(missing)} unexpected={len(unexpected)}"
        )
        raise RuntimeError(
            "transfer_e3_gin_to_xa strict failure: "
            f"missing={missing} unexpected={unexpected}. "
            "Do not fall back to strict=False without audit."
        )

    # Apply only audited keys
    new_state = {**target_state, **loaded}
    gin.load_state_dict(new_state, strict=True)

    transferred = {k: gin.state_dict()[k].detach().cpu() for k in loaded}
    report.target_hash = _tensor_sha(transferred)
    report.ok = len(missing) == 0
    report.notes.append(
        f"loaded {len(loaded)} convs/bns keys; ignored {len(ignored)} pool/proj keys"
    )
    return report


def write_transfer_report(path: Path, report: GinTransferReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def transfer_e3_gin_into_module(
    predictive_checkpoint: Path | str,
    gin: nn.Module,
    *,
    strict: bool = True,
) -> GinTransferReport:
    """Transfer into a bare GINConvNet (Predictive encoder or XA.drug_encoder.gin)."""

    class _Adapter(nn.Module):
        def __init__(self, g: nn.Module) -> None:
            super().__init__()
            self.drug_encoder = nn.Module()
            self.drug_encoder.gin = g  # type: ignore[attr-defined]

    return transfer_e3_gin_to_xa(predictive_checkpoint, _Adapter(gin), strict=strict)


def verify_transferred_weights_match(
    predictive_checkpoint: Path | str,
    xa_model: nn.Module,
    *,
    keys: Optional[Sequence[str]] = None,
) -> bool:
    ckpt = torch.load(Path(predictive_checkpoint), map_location="cpu")
    source = _extract_encoder_state(ckpt)
    gin_state = xa_model.drug_encoder.gin.state_dict()
    check_keys = list(keys) if keys is not None else [k for k in source if k.startswith(TRANSFER_PREFIXES)]
    for key in check_keys:
        if key not in gin_state:
            return False
        if not torch.allclose(gin_state[key].cpu(), source[key].cpu()):
            return False
    return True
