"""Load / save Stage2 AE checkpoints for Round 25 parity."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


AE_FILE_MAP = {
    "shared_vae": "after_pretrain_shared_vae.pth",
    "source_vae": "after_pretrain_source_vae.pth",
    "target_vae": "after_pretrain_target_vae.pth",
    "classifier": "after_pretrain_classifier.pth",
}


def save_ae_checkpoint(exp_dir: str | Path, modules: Dict[str, torch.nn.Module]) -> Dict[str, str]:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key, fname in AE_FILE_MAP.items():
        path = exp_dir / fname
        torch.save(modules[key].state_dict(), path)
        saved[key] = str(path)
    return saved


def load_ae_checkpoint(
    exp_dir: str | Path,
    modules: Dict[str, torch.nn.Module],
    *,
    map_location="cpu",
    allow_traingan_fallback: bool = False,
) -> Dict[str, str]:
    """Load AE weights. Prefer after_pretrain_*; optionally fall back to after_traingan_*."""
    exp_dir = Path(exp_dir)
    loaded = {}
    fallback_map = {
        "shared_vae": "after_traingan_shared_vae.pth",
        "source_vae": "after_traingan_source_vae.pth",
        "target_vae": "after_traingan_target_vae.pth",
        "classifier": "after_traingan_classifier.pth",
    }
    for key, fname in AE_FILE_MAP.items():
        path = exp_dir / fname
        if not path.exists() and allow_traingan_fallback:
            path = exp_dir / fallback_map[key]
        if not path.exists():
            raise FileNotFoundError(f"missing AE checkpoint for {key}: {path}")
        state = torch.load(path, map_location=map_location)
        modules[key].load_state_dict(state)
        loaded[key] = str(path)
    return loaded
