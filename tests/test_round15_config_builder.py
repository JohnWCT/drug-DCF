import os
import pandas as pd
import pytest

from tools.round15_config_builder import (
    COMPACT_FEATURE_MODES,
    build_round15_all,
    build_round15_pretrain_configs,
    resolve_round15b_model_pool,
)
from tools.round9_diagnostics_common import load_json, resolve_path

SETTINGS = "config/round15_repro_rescue_settings.json"
SMOKE_OUT = "result/optimization_runs/round15_repro_rescue_test_build"


@pytest.fixture(scope="module")
def settings():
    return load_json(resolve_path(SETTINGS))


def test_reads_settings(settings):
    assert settings["round"] == "round15"
    assert "round15a_repro" in settings


def test_build_15a_15b_15c_manifests(tmp_path_factory):
    outdir = resolve_path(SMOKE_OUT)
    build_round15_all(SETTINGS, outdir, force=True)
    pretrain = pd.read_csv(os.path.join(outdir, "manifests/pretrain_sweep_manifest.csv"))
    finetune = pd.read_csv(os.path.join(outdir, "manifests/finetune_dispatch_manifest.csv"))
    assert len(pretrain) == 24
    assert set(finetune["feature_mode"].unique()) <= set(COMPACT_FEATURE_MODES)
    assert (finetune["round15_branch"] == "A").sum() == 40
    assert "exp_008" in finetune["source_model_id"].astype(str).values


def test_15b_pool_includes_exp008(settings):
    pool, _ = resolve_round15b_model_pool(settings)
    ids = {p["pool_model_id"] for p in pool}
    assert "exp_008" in ids
    assert "exp_035" in ids


def test_pretrain_only_flag():
    outdir = resolve_path(SMOKE_OUT)
    path = build_round15_pretrain_configs(SETTINGS, outdir, force=True)
    assert path and os.path.isfile(path)
