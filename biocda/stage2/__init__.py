"""Round 25 Stage 2 package."""

from biocda.stage2.latent_autoencoder import LatentAutoencoder
from biocda.stage2.target_adapter import TargetResidualAdapter
from biocda.stage2.variant_registry import (
    build_variant_registry,
    initial_screen_variants,
    load_registry_from_yaml,
    registry_payload,
)
from biocda.stage2.variants import FIXED_XA_CONTRACT, Stage2Variant

__all__ = [
    "LatentAutoencoder",
    "TargetResidualAdapter",
    "Stage2Variant",
    "FIXED_XA_CONTRACT",
    "build_variant_registry",
    "load_registry_from_yaml",
    "registry_payload",
    "initial_screen_variants",
]
