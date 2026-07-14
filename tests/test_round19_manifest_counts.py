"""Tests for Round 19 config builder / job metadata."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.round19_config_builder import build_stage19b_manifest
from tools.round19_fusion_models import COMPATIBLE_CELLS
from tools.round19_manifest_validator import assert_expected_job_count, validate_compatible_manifest


@pytest.fixture
def settings():
    return json.loads(Path("config/round19_factorial_settings.json").read_text())


def test_build_stage19b_manifest_count(settings, tmp_path):
    df = build_stage19b_manifest(settings, str(tmp_path), omics_ids=["O1", "O3"], n_folds=3)
    validate_compatible_manifest(df)
    assert_expected_job_count(df, len(COMPATIBLE_CELLS) * 2 * 3)
    assert df["job_id"].is_unique
    assert "node_hidden_dim" in df.columns
    assert "split_seed" in df.columns
