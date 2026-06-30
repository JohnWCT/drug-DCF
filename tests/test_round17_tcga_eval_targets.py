"""Round 17 Phase 0: five-target TCGA eval configuration."""

from __future__ import annotations

import os

import pytest

from tools.finetune_tcga_eval import (
    DEFAULT_TCGA_EVAL_PREFIX_MAP,
    DEFAULT_TCGA_EVAL_TARGETS,
    EVAL_KEY_TO_LEGACY_TAG,
    FIXED_DRUG_SMILES_AACDR_EXTENDED,
    FIXED_TCGA_EVAL_AACDR_GDSC_INTERSECT,
    FIXED_TCGA_EVAL_AACDR_TCGA_ONLY,
    HISTORICAL_TCGA_EVAL_KEYS,
    ROUND17_TCGA_EVAL_KEYS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_round17_has_five_tcga_targets():
    keys = [key for key, _ in DEFAULT_TCGA_EVAL_TARGETS]
    assert keys == [
        "gdsc_intersect13",
        "tcga_only3",
        "dapl",
        "aacdr_tcga_only",
        "aacdr_gdsc_intersect",
    ]


def test_round17_eval_target_paths():
    path_by_key = dict(DEFAULT_TCGA_EVAL_TARGETS)
    assert path_by_key["aacdr_tcga_only"] == FIXED_TCGA_EVAL_AACDR_TCGA_ONLY
    assert path_by_key["aacdr_gdsc_intersect"] == FIXED_TCGA_EVAL_AACDR_GDSC_INTERSECT


def test_round17_legacy_tags_include_aacdr():
    assert EVAL_KEY_TO_LEGACY_TAG["aacdr_tcga_only"] == "AACDR_TCGA_ONLY"
    assert EVAL_KEY_TO_LEGACY_TAG["aacdr_gdsc_intersect"] == "AACDR_GDSC_INTERSECT"


def test_round17_prefix_map_includes_aacdr():
    assert DEFAULT_TCGA_EVAL_PREFIX_MAP["aacdr_tcga_only"] == "aacdr_tcga_only_"
    assert DEFAULT_TCGA_EVAL_PREFIX_MAP["aacdr_gdsc_intersect"] == "aacdr_gdsc_intersect_"


def test_historical_keys_remain_three_target_subset():
    assert HISTORICAL_TCGA_EVAL_KEYS == ("gdsc_intersect13", "tcga_only3", "dapl")
    assert ROUND17_TCGA_EVAL_KEYS == (
        "gdsc_intersect13",
        "tcga_only3",
        "dapl",
        "aacdr_tcga_only",
        "aacdr_gdsc_intersect",
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        FIXED_TCGA_EVAL_AACDR_TCGA_ONLY,
        FIXED_TCGA_EVAL_AACDR_GDSC_INTERSECT,
        FIXED_DRUG_SMILES_AACDR_EXTENDED,
    ],
)
def test_round17_data_files_exist(rel_path):
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    assert os.path.isfile(full_path), f"missing Round 17 data file: {rel_path}"
