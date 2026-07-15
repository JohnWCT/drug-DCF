#!/usr/bin/env python3
"""Build a deterministic Round 19 model card without inventing a 19G verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.round19_reproducibility_audit import (
    attach_canonical_hash,
    sha256_file,
    write_json,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _extract_19g(report: Mapping[str, Any]) -> dict[str, Any]:
    verdict = report.get("verdict")
    if verdict is None:
        verdict = report.get("status")
    if not isinstance(verdict, str) or verdict in {"", "awaiting_19g"}:
        raise ValueError("A supplied 19G report must contain an explicit final verdict/status")
    return {
        "status": "complete",
        "verdict": verdict,
        "report_stage": report.get("stage", "19g"),
        "report_canonical_sha256": report.get("canonical_sha256"),
        "summary": report.get("summary"),
    }


def build_model_card(
    final_lock_path: Path,
    *,
    report_19g_path: Path | None = None,
) -> dict[str, Any]:
    lock = _read_object(final_lock_path)
    if lock.get("lock_type") != "round19_final_role_lock":
        raise AssertionError("Model card requires the immutable Round 19 final role lock")
    inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
    evaluation = (
        _extract_19g(_read_object(report_19g_path))
        if report_19g_path is not None
        else {
            "status": "awaiting_19g",
            "verdict": None,
            "summary": None,
            "statement": "No Stage 19G result was supplied; no verdict is asserted.",
        }
    )
    payload = {
        "schema": "round19_model_card",
        "schema_version": 1,
        "stage": "19h",
        "all_done": False,
        "status": evaluation["status"],
        "model_identity": {
            "policy": "multi_role_no_single_champion",
            "single_champion": lock.get("single_champion"),
            "roles": lock.get("roles", {}),
            "locked_checkpoint_count": len(inventory),
            "final_role_lock_sha256": sha256_file(final_lock_path),
            "lock_payload_sha256": lock.get("hashes", {}).get("lock_payload_sha256"),
        },
        "evaluation_19g": evaluation,
        "limitations": [
            "Post-hoc internal and TCGA findings must not alter locked model roles.",
            "No Stage 19H completion claim is valid until Stage 19G is complete.",
        ],
    }
    return attach_canonical_hash(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--report-19g")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    card = build_model_card(
        Path(args.final_lock),
        report_19g_path=Path(args.report_19g) if args.report_19g else None,
    )
    if args.output and not args.dry_run:
        write_json(Path(args.output), card)
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
