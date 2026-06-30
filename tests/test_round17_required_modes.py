"""Verify Round 17 settings feature modes are implemented in prototype_response_features."""

from __future__ import annotations

from tools.prototype_response_features import (
    SUPPORTED_MODES,
    compute_round17_standalone_features_batch,
    get_projected_context_dim,
    get_projected_delta_dim,
    is_round17_standalone_mode,
)

REQUIRED_ROUND17_MODES = [
    "own_proto_delta_projected_8",
    "own_proto_delta_projected_16",
    "own_proto_delta_projected_32",
    "own_proto_delta_projected_64",
    "own_proto_context_projected_16",
    "own_proto_context_projected_32",
    "own_plus_summary_plus_delta_projected_16",
    "source_proto_delta_projected_16",
    "target_available_context_projected_16",
    "minimal_source_only_min_margin",
]


def test_round17_settings_modes_all_supported():
    missing = [m for m in REQUIRED_ROUND17_MODES if m not in SUPPORTED_MODES]
    assert not missing, f"missing from SUPPORTED_MODES: {missing}"


def test_round17_projection_dims():
    assert get_projected_delta_dim("own_proto_delta_projected_8") == 8
    assert get_projected_delta_dim("own_proto_delta_projected_64") == 64
    assert get_projected_context_dim("target_available_context_projected_16") == 16


def test_round17_standalone_helpers_importable():
    assert is_round17_standalone_mode("minimal_source_only_min_margin")
    assert callable(compute_round17_standalone_features_batch)
