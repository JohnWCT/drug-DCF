"""Round 25 loss package — keep margin field names distinct."""

from biocda.losses.aada_reconstruction import (
    AADADiscriminatorOutput,
    AADATargetAdapterOutput,
    aada_ae_discriminator_loss,
    aada_target_adapter_loss,
)
from biocda.losses.prototype_band import PrototypeBandOutput, prototype_distance_band_loss
from biocda.losses.prototype_margin import PrototypeMarginOutput, margin_gated_prototype_loss
from biocda.losses.smooth_l1_vector import vector_smooth_l1

__all__ = [
    "PrototypeMarginOutput",
    "margin_gated_prototype_loss",
    "PrototypeBandOutput",
    "prototype_distance_band_loss",
    "vector_smooth_l1",
    "AADADiscriminatorOutput",
    "AADATargetAdapterOutput",
    "aada_ae_discriminator_loss",
    "aada_target_adapter_loss",
]
