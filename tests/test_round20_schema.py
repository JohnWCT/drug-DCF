from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.round20_schema import Round20SchemaError, normalize_dimensions, validate_settings

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config/round20_unseen_drug_closure_settings.json"
GUARDRAILS = ROOT / "config/round20_guardrails.json"


def test_normalize_dimensions_accepts_unsorted() -> None:
    assert normalize_dimensions([32, 16]) == [16, 32]


def test_normalize_dimensions_rejects_other() -> None:
    with pytest.raises(Round20SchemaError):
        normalize_dimensions([8, 16])


def test_repo_settings_pass_preflight_schema() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    guardrails = json.loads(GUARDRAILS.read_text(encoding="utf-8"))
    report = validate_settings(settings, guardrails=guardrails, require_feature_dirs=True)
    assert report["ok"] is True
    assert report["split_seeds"] == [52, 62, 72]
    assert report["feature_dirs_present"]["16"] is True
    assert report["feature_dirs_present"]["32"] is True


def test_require_feature_dirs_fails_when_c32_missing() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    settings = json.loads(json.dumps(settings))
    settings["omics"]["feature_dirs"]["32"] = None
    with pytest.raises(Round20SchemaError, match=r"feature_dirs\[32\]"):
        validate_settings(settings, require_feature_dirs=True)
