"""Tests for Stage 19C manifest builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.round19_config_builder import build_stage19c_manifest


@pytest.fixture
def settings(tmp_path):
    return {
        "screening_split_seed": 42,
        "model_seed": 101,
        "round19_feature_out_root": str(tmp_path / "features"),
        "drug_reps": {
            "D0": {"type": "gin", "node_hidden_dim": 32, "graph_output_dim": 32, "edge_features": False},
            "D2": {"type": "gin", "node_hidden_dim": 64, "graph_output_dim": 64, "edge_features": False},
            "D4": {"type": "maccs", "output_dim": 64},
        },
    }


@pytest.fixture
def candidate_lock():
    return {
        "lock_type": "stage19c_candidate_lock",
        "unique_cells": [
            {"drug_id": "D0", "predictor_id": "P0", "primary_role": "R0"},
            {"drug_id": "D0", "predictor_id": "P2", "primary_role": "R2"},
            {"drug_id": "D2", "predictor_id": "P0", "primary_role": "R3"},
            {"drug_id": "D4", "predictor_id": "P1", "primary_role": "R6"},
        ],
        "context_shuffle_controls": {
            "atom_cell": {"drug_id": "D0", "predictor_id": "P2"},
            "pooled_cell": {"drug_id": "D0", "predictor_id": "P0"},
        },
        "best_pooled_for_shuffle": {"drug_id": "D0", "predictor_id": "P0"},
    }


def test_manifest_counts(settings, candidate_lock, tmp_path):
    outdir = str(tmp_path / "round19")
    df = build_stage19c_manifest(settings, outdir, candidate_lock, include_context_controls=True)
    n_selected = 4
    core = df[df["control_type"] == "none"]
    ctrl = df[df["control_type"] == "context_shuffle"]
    assert len(core) == n_selected * 2 * 3
    assert len(ctrl) == 12
    assert len(df) == n_selected * 2 * 3 + 12
    assert df["job_id"].is_unique
    assert df["result_dir"].is_unique


def test_core_omics_folds(settings, candidate_lock, tmp_path):
    outdir = str(tmp_path / "round19")
    df = build_stage19c_manifest(settings, outdir, candidate_lock, include_context_controls=True)
    core = df[df["control_type"] == "none"]
    for (_, _), group in core.groupby(["drug_id", "predictor_id"]):
        assert set(group["omics_id"]) == {"O0", "O4"}
        for omics_id in ("O0", "O4"):
            folds = set(group.loc[group["omics_id"] == omics_id, "fold_id"].astype(int))
            assert folds == {0, 1, 2}


def test_shuffle_rows_have_seeds(settings, candidate_lock, tmp_path):
    outdir = str(tmp_path / "round19")
    df = build_stage19c_manifest(settings, outdir, candidate_lock, include_context_controls=True)
    ctrl = df[df["control_type"] == "context_shuffle"]
    assert set(ctrl["omics_id"]) == {"O2", "O3"}
    assert ctrl["train_shuffle_seed"].notna().all()
    assert ctrl["validation_shuffle_seed"].notna().all()
