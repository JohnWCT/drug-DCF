"""Stage 2 variant contracts for Round 25."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Stage2Variant:
    variant_id: str
    global_alignment: str  # wgan | aada_autoencoder
    conditional_alignment: str  # conditional_wgan
    prototype_alignment: str  # always_on | margin_gated | distance_band
    lock_eligible: bool = True
    enabled_if: Optional[Dict[str, Any]] = None
    prototype_margin: Optional[Dict[str, Any]] = None
    prototype_band: Optional[Dict[str, Any]] = None
    aada: Optional[Dict[str, Any]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REQUIRED_VARIANTS = ("S0", "S2", "S1")
CONDITIONAL_VARIANTS = ("S3", "S2b")

# Fixed downstream XA — not searched in Round 25.
FIXED_XA_CONTRACT = {
    "latent": "Z64",
    "context": "C32",
    "query": "single_omics_context_token",
    "drug_encoder": "fresh_gin_atom",
    "graph_pooling": False,
    "attention": "sample_to_atom_cross_attention",
    "head": "final_updated_query",
    "forbid_e3_transfer": True,
    "forbid_kd": True,
    "summary_dim": 0,
}
