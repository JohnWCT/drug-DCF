import copy
import json
import os
import tempfile

import pandas as pd

from tools.optimization_config_generator import (
    MANIFEST_COLUMNS,
    expand_sweep_combinations,
    generate_configs,
)


def test_generates_exactly_72_configs(tmp_path):
    spec = {
        "base_config": "config/params_proto_base_vaewc.json",
        "output_config_dir": str(tmp_path / "generated"),
        "run_id": "test_round1",
        "sweep": {
            "lambda_proto": [0, 0.03, 0.1, 0.3],
            "proto_temperature": [0.1, 0.2, 0.5],
            "proto_start_epoch": [5, 10, 30],
            "proto_full_epoch": [30, 50],
            "lambda_adv": [1.0],
            "gan_gen_update_interval": [5],
        },
    }
    spec_path = tmp_path / "sweep.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    manifest_path, manifest_df = generate_configs(str(spec_path), manifest_dir=str(tmp_path / "manifests"), force=True)
    assert len(manifest_df) == 72
    assert len(list((tmp_path / "generated").glob("*.json"))) == 72
    assert (manifest_df["lambda_proto"] == 0).sum() == 18
    assert list(manifest_df.columns) == MANIFEST_COLUMNS


def test_does_not_mutate_base_object(tmp_path):
    base = {"pretrain_params": {"lambda_cls": [10], "lambda_proto": [999]}}
    original = copy.deepcopy(base)
    combos = expand_sweep_combinations({"lambda_proto": [0, 0.1]})
    assert base == original
    assert len(combos) == 2


def test_force_overwrite(tmp_path):
    spec = {
        "base_config": "config/params_proto_base_vaewc.json",
        "output_config_dir": str(tmp_path / "generated"),
        "run_id": "force_test",
        "sweep": {"lambda_proto": [0], "proto_temperature": [0.2], "proto_start_epoch": [5], "proto_full_epoch": [30], "lambda_adv": [1.0], "gan_gen_update_interval": [5]},
    }
    spec_path = tmp_path / "sweep.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    manifest_path, _ = generate_configs(str(spec_path), manifest_dir=str(tmp_path / "manifests"), force=True)
    first_mtime = os.path.getmtime(manifest_path)
    _, df2 = generate_configs(str(spec_path), manifest_dir=str(tmp_path / "manifests"), force=True)
    assert len(df2) == 1
    assert os.path.getmtime(manifest_path) >= first_mtime
