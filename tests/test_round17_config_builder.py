"""Round 17 config builder smoke."""

from __future__ import annotations

import json
import os
import tempfile

from tools.round17_direct_proto_config_builder import build_round17_configs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_round17_config_builder_stage17a_smoke():
    settings_path = os.path.join(PROJECT_ROOT, "config/round17_direct_proto_settings.json")
    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)
    settings["stage17a"]["models"] = ["r13_exp_008"]
    settings["stage17a"]["feature_modes"] = ["none", "own_proto_delta_projected_16"]
    settings["stage17a"]["seeds"] = [101]
    settings["stage17a"]["max_combos"] = 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_settings = os.path.join(tmp, "settings.json")
        with open(tmp_settings, "w", encoding="utf-8") as f:
            json.dump(settings, f)
        outdir = os.path.join(tmp, "round17")
        outputs = build_round17_configs(tmp_settings, outdir, "17a")
        assert outputs["n_jobs"] == 2
        assert os.path.isfile(os.path.join(outdir, "manifests/stage17a_proto_feature_manifest.csv"))
        assert os.path.isfile(os.path.join(outdir, "manifests/stage17a_finetune_dispatch_manifest.csv"))
