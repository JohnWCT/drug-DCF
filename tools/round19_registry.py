"""Round 19 O/D/P registries and compatibility matrix.

Source of truth for the local factorial run is
``config/round19_factorial_settings.json`` plus
``tools.round19_fusion_models.COMPATIBLE_CELLS``.  Public follow-up manuals may
describe an idealized D4×P0-only matrix; this registry documents the *executed*
13-cell matrix without rewriting historical jobs.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.round19_fusion_models import (  # noqa: F401
    COMPATIBLE_CELLS as FUSION_COMPATIBLE_CELLS,
    assert_compatible,
)
from tools.round19_schema import DRUG_IDS, OMICS_IDS, PREDICTOR_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "round19_factorial_settings.json"

# Manual-ideal matrix (not the local executed matrix).
PUBLIC_MANUAL_COMPATIBLE_CELLS: list[tuple[str, str]] = [
    (drug_id, predictor_id)
    for drug_id in ("D0", "D1", "D2", "D3")
    for predictor_id in ("P0", "P1", "P2")
] + [("D4", "P0")]


@lru_cache(maxsize=4)
def load_settings(path: str | None = None) -> dict[str, Any]:
    settings_path = Path(path) if path else DEFAULT_SETTINGS
    value = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Settings must be a JSON object: {settings_path}")
    return value


def build_omics_registry(settings: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    cfg = settings or load_settings()
    omics = cfg.get("omics_ids", {})
    out: dict[str, dict[str, Any]] = {}
    for omics_id in OMICS_IDS:
        record = omics.get(omics_id, {})
        out[omics_id] = {
            "name": str(record.get("display_name", omics_id)),
            "expected_dim": record.get("dim"),
        }
    return out


def build_drug_registry(settings: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    cfg = settings or load_settings()
    drugs = cfg.get("drug_reps", {})
    out: dict[str, dict[str, Any]] = {}
    for drug_id in DRUG_IDS:
        record = drugs.get(drug_id, {})
        family = str(record.get("type", "")).lower()
        out[drug_id] = {
            "family": family,
            "node_dim": record.get("node_hidden_dim"),
            "graph_dim": record.get("graph_output_dim", record.get("output_dim")),
            "bond_aware": bool(record.get("edge_features", False)),
            "edge_dim": record.get("edge_dim"),
            "input_dim": record.get("input_dim"),
        }
    return out


def build_predictor_registry(settings: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    cfg = settings or load_settings()
    predictors = cfg.get("predictors", {})
    families = {
        "P0": "pooled_mlp",
        "P1": "pooled_transformer",
        "P2": "atom_cross_attention",
    }
    out: dict[str, dict[str, Any]] = {}
    for predictor_id in PREDICTOR_IDS:
        out[predictor_id] = {
            "family": families[predictor_id],
            "canonical_name": str(predictors.get(predictor_id, families[predictor_id])),
            "requires_nodes": predictor_id == "P2",
        }
    return out


def build_compatible_cells(settings: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """Return the executed local compatibility matrix (authoritative)."""
    cfg = settings or load_settings()
    raw = cfg.get("compatible_cells")
    if raw:
        cells = [(str(drug), str(pred)) for drug, pred in raw]
    else:
        cells = list(FUSION_COMPATIBLE_CELLS)
    if len(cells) != 13:
        raise AssertionError(f"Expected 13 compatible cells, got {len(cells)}")
    # Keep fusion_models as runtime authority for training paths.
    fusion = set(FUSION_COMPATIBLE_CELLS)
    settings_set = set(cells)
    if fusion != settings_set:
        raise AssertionError(
            "Settings compatible_cells drift from fusion_models.COMPATIBLE_CELLS: "
            f"only_settings={sorted(settings_set - fusion)} "
            f"only_fusion={sorted(fusion - settings_set)}"
        )
    return cells


def assert_compatible_cell(drug_id: str, predictor_id: str) -> None:
    assert_compatible(str(drug_id), str(predictor_id))


OMICS_REGISTRY = build_omics_registry()
DRUG_REGISTRY = build_drug_registry()
PREDICTOR_REGISTRY = build_predictor_registry()
# Public follow-up manual import path; equals the executed local matrix.
COMPATIBLE_CELLS: list[tuple[str, str]] = build_compatible_cells()


def registry_snapshot() -> dict[str, Any]:
    return {
        "schema": "round19_registry_snapshot",
        "schema_version": 1,
        "source_settings": str(DEFAULT_SETTINGS.relative_to(PROJECT_ROOT)),
        "omics": OMICS_REGISTRY,
        "drugs": DRUG_REGISTRY,
        "predictors": PREDICTOR_REGISTRY,
        "compatible_cells": [list(cell) for cell in COMPATIBLE_CELLS],
        "compatible_cell_count": len(COMPATIBLE_CELLS),
        "public_manual_compatible_cells": [
            list(cell) for cell in PUBLIC_MANUAL_COMPATIBLE_CELLS
        ],
        "note": (
            "Local factorial executed D4×P1 and omitted D1×P2; "
            "public manual ideal is D4×P0-only with full D0–D3×P0–P2."
        ),
    }


def validate_registry_invariants(
    *,
    settings: Mapping[str, Any] | None = None,
    cells: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    cells = list(cells) if cells is not None else build_compatible_cells(settings)
    if len(cells) != 13:
        raise AssertionError(f"Expected exactly 13 cells, got {len(cells)}")
    if len(set(cells)) != len(cells):
        raise AssertionError("Duplicate drug×predictor cells")
    for drug_id, predictor_id in cells:
        if drug_id not in DRUG_IDS or predictor_id not in PREDICTOR_IDS:
            raise AssertionError(f"Unknown cell member: {drug_id}×{predictor_id}")
        assert_compatible_cell(drug_id, predictor_id)
    # Local executed matrix forbids D4×P2.
    if ("D4", "P2") in cells:
        raise AssertionError("D4×P2 must remain incompatible")
    return {
        "ok": True,
        "cell_count": len(cells),
        "includes_d4_p1": ("D4", "P1") in cells,
        "includes_d1_p2": ("D1", "P2") in cells,
    }
