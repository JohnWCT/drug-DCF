#!/usr/bin/env python3
"""Thin Round 19 selection CLI wrapping existing stage selectors.

This adapter never reads internal/TCGA selection scores and never invents
new winners.  It only exposes a stable public entrypoint over stage modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.round19_schema import (
    ALLOWED_SELECTION_ATTESTATION_KEYS,
    validate_selection_input_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = PROJECT_ROOT / "result/optimization_runs/round19_factorial"


def _refuse_training_selection(from_stage: str, to_stage: str) -> dict[str, Any]:
    return {
        "schema": "round19_selection_adapter",
        "schema_version": 1,
        "status": "adapter_only",
        "from_stage": from_stage,
        "to_stage": to_stage,
        "message": (
            "Round 19 selection locks already exist under stage selectors. "
            "This adapter verifies artifacts and forbids re-selection."
        ),
        "existing_modules": {
            "19b_to_19c": "tools.round19_stage19c_selector",
            "19c_to_19d": "tools.round19_stage19d_selector",
            "19d_to_19e": "tools.round19_stage19e_selector",
            "19e_to_19f": "tools.round19_stage19f_role_selector",
        },
    }


def verify_selection_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Selection artifact must be an object: {path}")
    validate_selection_input_columns(payload.keys())

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            validate_selection_input_columns(node.keys())
            for key, value in node.items():
                if key in ALLOWED_SELECTION_ATTESTATION_KEYS and key.endswith("_used"):
                    if value not in (False, None, 0):
                        raise AssertionError(
                            f"Attestation flag {trail}.{key} must be false/null, got {value!r}"
                        )
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")

    walk(payload, "root")
    return {
        "path": str(path),
        "ok": True,
        "top_level_keys": sorted(payload.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", required=True)
    parser.add_argument("--to-stage", required=True)
    parser.add_argument(
        "--verify-artifact",
        action="append",
        default=[],
        help="Existing selection lock/proposal JSON to verify for leakage",
    )
    parser.add_argument("--allow-reselect", action="store_true")
    args = parser.parse_args()
    if args.allow_reselect:
        raise SystemExit(
            "Re-selection is disabled in the public-reconstruction adapter; "
            "use the original stage selector modules only under an explicit new Round."
        )
    report = _refuse_training_selection(args.from_stage, args.to_stage)
    report["verified_artifacts"] = [
        verify_selection_artifact(Path(path)) for path in args.verify_artifact
    ]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
