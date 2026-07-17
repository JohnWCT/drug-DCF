#!/usr/bin/env python3
"""Round 20 settings / contract schema validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = PROJECT_ROOT / "config/round20_unseen_drug_closure_settings.json"
DEFAULT_GUARDRAILS = PROJECT_ROOT / "config/round20_guardrails.json"

REQUIRED_TOP_LEVEL = (
    "round",
    "purpose",
    "round19_role_lock",
    "round19_deployment_policy",
    "result_root",
    "selection_data_scope",
    "omics",
    "drug",
    "predictors",
    "validation",
    "selection",
    "tcga",
)

FORBIDDEN_SELECTION_TOKENS = (
    "tcga",
    "tcga_only",
    "internal_test",
    "external",
    "integrated5",
    "posthoc",
    "cancer_held_out",
    "patient",
    "target_cohort",
)

PLACEHOLDER_MARKERS = (
    "<C16_FEATURE_DIR>",
    "<C32_FEATURE_DIR>",
    "<OMICS_ENCODER_CHECKPOINT>",
    "<GENE_ORDER_ARTIFACT>",
    "<OMICS_NORMALIZATION_ARTIFACT>",
)


class Round20SchemaError(ValueError):
    """Raised when Round 20 settings violate the immutable research contract."""


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Round20SchemaError(f"Expected JSON object at {p}")
    return payload


def _require_keys(obj: Mapping[str, Any], keys: Sequence[str], *, where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise Round20SchemaError(f"{where} missing keys: {missing}")


def _as_int_list(value: Any, *, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise Round20SchemaError(f"{where} must be a non-empty list")
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            raise Round20SchemaError(f"{where} must contain integers") from exc
    return out


def normalize_dimensions(dimensions: Sequence[int]) -> list[int]:
    dims = sorted({int(d) for d in dimensions})
    if dims != [16, 32]:
        raise Round20SchemaError(
            f"omics.dimensions must normalize to [16, 32], got {list(dimensions)} -> {dims}"
        )
    return dims


def find_placeholders(obj: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            hits.extend(find_placeholders(value, path=f"{path}.{key}"))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(find_placeholders(value, path=f"{path}[{idx}]"))
    elif isinstance(obj, str):
        for marker in PLACEHOLDER_MARKERS:
            if marker in obj:
                hits.append(f"{path}={obj}")
    return hits


def validate_drug_contract(drug: Mapping[str, Any]) -> None:
    _require_keys(
        drug,
        ("encoder_id", "input_dim", "hidden_dim", "num_layers", "jk_mode", "pool", "training_mode"),
        where="drug",
    )
    if drug["encoder_id"] != "D0":
        raise Round20SchemaError(f"drug.encoder_id must be D0, got {drug['encoder_id']!r}")
    expected = {
        "input_dim": 78,
        "hidden_dim": 32,
        "num_layers": 5,
        "jk_mode": "last",
        "pool": "max",
    }
    for key, want in expected.items():
        got = drug[key]
        if key in ("input_dim", "hidden_dim", "num_layers"):
            got = int(got)
        if got != want:
            raise Round20SchemaError(f"drug.{key} must be {want!r}, got {got!r}")


def validate_selection_contract(selection: Mapping[str, Any], guardrails: Mapping[str, Any] | None = None) -> None:
    _require_keys(
        selection,
        (
            "primary_metric",
            "tie_breaker_metric",
            "parsimony_delta",
            "auprc_max_drop",
            "major_fail_auc_delta",
            "min_nonworse_seeds",
        ),
        where="selection",
    )
    if selection["primary_metric"] != "DrugMacro_AUC":
        raise Round20SchemaError("selection.primary_metric must be DrugMacro_AUC")
    if selection["tie_breaker_metric"] != "DrugMacro_AUPRC":
        raise Round20SchemaError("selection.tie_breaker_metric must be DrugMacro_AUPRC")
    if float(selection["parsimony_delta"]) != 0.005:
        raise Round20SchemaError("selection.parsimony_delta must be 0.005")
    if float(selection["auprc_max_drop"]) != 0.01:
        raise Round20SchemaError("selection.auprc_max_drop must be 0.01")
    if float(selection["major_fail_auc_delta"]) != -0.02:
        raise Round20SchemaError("selection.major_fail_auc_delta must be -0.02")
    if int(selection["min_nonworse_seeds"]) != 2:
        raise Round20SchemaError("selection.min_nonworse_seeds must be 2")
    if guardrails is not None:
        rules = guardrails.get("rules", {})
        if int(rules.get("G2_min_nonworse_seeds", 2)) != 2:
            raise Round20SchemaError("guardrails G2_min_nonworse_seeds must be 2")
        if float(rules.get("G3_auprc_max_drop", 0.01)) != 0.01:
            raise Round20SchemaError("guardrails G3_auprc_max_drop must be 0.01")
        if float(rules.get("G4_major_fail_auc_delta", -0.02)) != -0.02:
            raise Round20SchemaError("guardrails G4_major_fail_auc_delta must be -0.02")


def validate_settings(
    settings: Mapping[str, Any],
    *,
    require_resolved_placeholders: bool = False,
    require_feature_dirs: bool = False,
    guardrails: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_keys(settings, REQUIRED_TOP_LEVEL, where="settings")
    if settings.get("round") != "round20":
        raise Round20SchemaError("round must be 'round20'")
    if settings.get("selection_data_scope") != "development_drug_held_out_only":
        raise Round20SchemaError(
            "selection_data_scope must be development_drug_held_out_only"
        )

    omics = settings["omics"]
    _require_keys(
        omics,
        ("latent_dim", "encoder_mode", "retain_end_to_end_path", "dimensions", "feature_dirs"),
        where="omics",
    )
    if int(omics["latent_dim"]) != 64:
        raise Round20SchemaError("omics.latent_dim must be 64")
    if omics.get("encoder_mode") != "frozen":
        raise Round20SchemaError("omics.encoder_mode must be frozen for Round 20")
    if omics.get("retain_end_to_end_path") is not True:
        raise Round20SchemaError("omics.retain_end_to_end_path must be true")
    dims = normalize_dimensions(_as_int_list(omics["dimensions"], where="omics.dimensions"))
    feature_dirs = omics["feature_dirs"]
    if not isinstance(feature_dirs, Mapping):
        raise Round20SchemaError("omics.feature_dirs must be an object")
    for dim in dims:
        key = str(dim)
        if key not in feature_dirs and dim not in feature_dirs:
            raise Round20SchemaError(f"omics.feature_dirs missing key for context dim {dim}")
        value = feature_dirs.get(key, feature_dirs.get(dim))
        if require_feature_dirs and not value:
            raise Round20SchemaError(f"omics.feature_dirs[{key}] is required but missing/null")

    validate_drug_contract(settings["drug"])

    predictors = settings["predictors"]
    if not isinstance(predictors, list) or set(predictors) != {
        "resolved_e3",
        "gated_pooled_fusion",
    }:
        raise Round20SchemaError(
            "predictors must be exactly ['resolved_e3', 'gated_pooled_fusion'] (order flexible)"
        )

    validation = settings["validation"]
    _require_keys(
        validation,
        ("split_type", "split_seeds", "n_splits", "model_seed", "group_column", "label_column"),
        where="validation",
    )
    if validation["split_type"] != "drug_held_out":
        raise Round20SchemaError("validation.split_type must be drug_held_out")
    seeds = _as_int_list(validation["split_seeds"], where="validation.split_seeds")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise Round20SchemaError("validation.split_seeds must be exactly three unique integers")
    if seeds != [52, 62, 72]:
        raise Round20SchemaError("validation.split_seeds must be [52, 62, 72]")
    if int(validation["n_splits"]) != 5:
        raise Round20SchemaError("validation.n_splits must be 5")
    if validation["group_column"] != "drug_group_id":
        raise Round20SchemaError("validation.group_column must be drug_group_id")
    if validation["label_column"] != "Label":
        raise Round20SchemaError("validation.label_column must be Label")

    validate_selection_contract(settings["selection"], guardrails=guardrails)

    tcga = settings["tcga"]
    if tcga.get("run_only_after_lock") is not True:
        raise Round20SchemaError("tcga.run_only_after_lock must be true")

    placeholders = find_placeholders(settings)
    if require_resolved_placeholders and placeholders:
        raise Round20SchemaError(f"Unresolved placeholders remain: {placeholders}")

    return {
        "ok": True,
        "dimensions": dims,
        "split_seeds": seeds,
        "placeholders": placeholders,
        "feature_dirs_present": {
            str(dim): bool(feature_dirs.get(str(dim), feature_dirs.get(dim)))
            for dim in dims
        },
    }


def validate_settings_file(
    settings_path: Path | str,
    *,
    guardrails_path: Path | str | None = None,
    require_resolved_placeholders: bool = False,
    require_feature_dirs: bool = False,
) -> dict[str, Any]:
    settings = load_json(settings_path)
    guardrails = None
    if guardrails_path is not None:
        guardrails = load_json(guardrails_path)
    elif settings.get("guardrails_config"):
        guardrails = load_json(PROJECT_ROOT / str(settings["guardrails_config"]))
    report = validate_settings(
        settings,
        require_resolved_placeholders=require_resolved_placeholders,
        require_feature_dirs=require_feature_dirs,
        guardrails=guardrails,
    )
    report["settings_path"] = str(settings_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS),
        help="Path to round20 settings JSON",
    )
    parser.add_argument("--guardrails", default=None)
    parser.add_argument(
        "--require-resolved-placeholders",
        action="store_true",
        help="Fail if angle-bracket placeholders remain",
    )
    parser.add_argument(
        "--require-feature-dirs",
        action="store_true",
        help="Fail if C16/C32 feature dirs are null/missing",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    report = validate_settings_file(
        args.settings,
        guardrails_path=args.guardrails,
        require_resolved_placeholders=args.require_resolved_placeholders,
        require_feature_dirs=args.require_feature_dirs,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
