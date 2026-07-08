#!/usr/bin/env python3
"""Round 17R config builder tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.round17r_18class_config_builder import (
    build_round17r_configs,
    _reject_forbidden_aliases,
)


def test_reject_round16_top_placeholder() -> None:
    with pytest.raises(ValueError, match="Forbidden Round 17R model alias"):
        _reject_forbidden_aliases("round16_top1")


def test_builder_stage_a_and_b(tmp_path: Path) -> None:
    settings_src = Path("config/round17r_18class_focused_settings.json")
    outdir = tmp_path / "round17r"
    try:
        a = build_round17r_configs(str(settings_src), str(outdir), "17r_a")
    except FileNotFoundError as exc:
        pytest.skip(f"checkpoints unavailable: {exc}")
    assert Path(a["stage17r_a_proto_feature_manifest"]).is_file()
    mana = pd.read_csv(a["stage17r_a_proto_feature_manifest"])
    assert "require_n_trainable_cancer_types" in mana.columns
    assert (mana["require_n_trainable_cancer_types"] == 18).all()

    b = build_round17r_configs(str(settings_src), str(outdir), "17r_b")
    manb = pd.read_csv(b["stage17r_b_finetune_dispatch_manifest"])
    assert len(manb) == 7 * 6 * 3
    assert "drug_smiles_path" in manb.columns
    assert manb["drug_smiles_path"].astype(str).str.contains("AACDR_extended").all()
    assert "finetune_config_path" in manb.columns
    assert "seed" in manb.columns
    assert "combo_id" in manb.columns
    assert "model_select_path" in manb.columns
    assert manb["model_select_path"].map(lambda p: Path(p).is_file()).all()
