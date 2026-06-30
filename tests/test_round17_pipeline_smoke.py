"""Round 17 pipeline import / compile smoke."""

from __future__ import annotations

import importlib


def test_round17_modules_importable():
    for mod in (
        "tools.round17_direct_proto_config_builder",
        "tools.analyze_round17_direct_proto",
        "tools.visualize_round17_prototype_tsne",
    ):
        importlib.import_module(mod)
