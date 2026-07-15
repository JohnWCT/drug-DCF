#!/usr/bin/env python3
"""D4-only MACCS bit ablation; results are fingerprint effects, never atom heatmaps."""
from __future__ import annotations

import torch


def ablate_maccs_bits(fingerprint: torch.Tensor, bit_indices, *, drug_role: str):
    if str(drug_role).upper() != "D4":
        raise ValueError("MACCS ablation is applicable only to the D4 role")
    out = fingerprint.clone()
    selected = sorted(set(map(int, bit_indices)))
    out[..., selected] = 0
    return out


def result_metadata(bit_indices) -> dict:
    return {
        "method": "maccs_bit_ablation",
        "bit_indices": sorted(set(map(int, bit_indices))),
        "attribution_level": "fingerprint_bit",
        "atom_heatmap_available": False,
    }
