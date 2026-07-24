"""Build / validate Stage 2 variant registry from Round 25 YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from biocda.stage2.variants import (
    CONDITIONAL_VARIANTS,
    FIXED_XA_CONTRACT,
    REQUIRED_VARIANTS,
    Stage2Variant,
)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML required to load Round 25 configs") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_variant_registry(config: Dict[str, Any]) -> Dict[str, Stage2Variant]:
    variants_cfg = dict(config.get("variants") or {})
    out: Dict[str, Stage2Variant] = {}
    for vid, raw in variants_cfg.items():
        raw = dict(raw or {})
        out[vid] = Stage2Variant(
            variant_id=vid,
            global_alignment=str(raw["global_alignment"]),
            conditional_alignment=str(raw["conditional_alignment"]),
            prototype_alignment=str(raw["prototype_alignment"]),
            lock_eligible=vid not in CONDITIONAL_VARIANTS or True,
            enabled_if=raw.get("enabled_if"),
            prototype_margin=raw.get("prototype_margin"),
            prototype_band=raw.get("prototype_band"),
            aada=raw.get("aada"),
        )
    missing = [v for v in REQUIRED_VARIANTS if v not in out]
    if missing:
        raise ValueError(f"missing required Stage25A variants: {missing}")
    return out


def load_registry_from_yaml(path: str | Path) -> Dict[str, Stage2Variant]:
    return build_variant_registry(_load_yaml(path))


def registry_payload(registry: Dict[str, Stage2Variant]) -> Dict[str, Any]:
    return {
        "fixed_xa": FIXED_XA_CONTRACT,
        "variants": {k: v.to_dict() for k, v in registry.items()},
        "required_screen_order": list(REQUIRED_VARIANTS),
        "conditional": list(CONDITIONAL_VARIANTS),
        "margin_field_names": {
            "prototype": ["prototype_upper_margin", "prototype_lower_margin"],
            "aada": ["reconstruction_margin"],
            "forbidden_alias": ["margin"],
        },
    }


def initial_screen_variants(registry: Dict[str, Stage2Variant]) -> List[str]:
    """S0/S2/S1 always; S3/S2b deferred to decision gates."""
    return [v for v in REQUIRED_VARIANTS if v in registry]
