from __future__ import annotations

import json
from pathlib import Path

from tools.round20_context_adapter import Round20O2FeatureStore, audit_context_pair
from tools.round20_schema import validate_settings

ROOT = Path(__file__).resolve().parents[1]
C16 = ROOT / "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context16"
C32 = ROOT / "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32"
SETTINGS = ROOT / "config/round20_unseen_drug_closure_settings.json"


def test_c16_c32_comparable_after_rebuild() -> None:
    assert C16.is_dir() and C32.is_dir()
    report = audit_context_pair(c16_dir=C16, c32_dir=C32, fail_closed=True)
    assert report["comparable"] is True
    assert report["mismatches"] == []


def test_feature_store_shapes() -> None:
    s16 = Round20O2FeatureStore(C16, 16)
    s32 = Round20O2FeatureStore(C32, 32)
    mid = next(iter(s16._load_mapping()))
    v16 = s16.get(mid)
    v32 = s32.get(mid)
    assert v16.shape == (80,)
    assert v32.shape == (96,)
    assert (v16[:64] == v32[:64]).all() or True  # Z may match if same latent source
    # Z should match exactly for shared ModelIDs
    assert (v16[:64] == v32[:64]).all()


def test_settings_require_feature_dirs_pass() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    report = validate_settings(settings, require_feature_dirs=True)
    assert report["ok"] is True
    assert report["feature_dirs_present"]["16"] is True
    assert report["feature_dirs_present"]["32"] is True
