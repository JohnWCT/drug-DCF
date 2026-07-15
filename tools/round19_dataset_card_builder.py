#!/usr/bin/env python3
"""Build a deterministic Round 19 dataset card from explicit dataset inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.round19_reproducibility_audit import (
    attach_canonical_hash,
    csv_fingerprint,
    write_json,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _evaluation(report_path: Path | None) -> dict[str, Any]:
    if report_path is None:
        return {
            "status": "awaiting_19g",
            "verdict": None,
            "statement": "No Stage 19G result was supplied; no dataset verdict is asserted.",
        }
    report = _read_object(report_path)
    verdict = report.get("verdict", report.get("status"))
    if not isinstance(verdict, str) or verdict in {"", "awaiting_19g"}:
        raise ValueError("A supplied 19G report must contain an explicit final verdict/status")
    return {
        "status": "complete",
        "verdict": verdict,
        "report_stage": report.get("stage", "19g"),
        "report_canonical_sha256": report.get("canonical_sha256"),
        "summary": report.get("summary"),
    }


def build_dataset_card(
    project_root: Path,
    datasets: Mapping[str, Path],
    *,
    report_19g_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    records = []
    for dataset_id, path in sorted(datasets.items()):
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        try:
            relative = absolute.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Dataset must be project-relative: {absolute}") from exc
        records.append(
            {
                "dataset_id": dataset_id,
                "path": relative,
                "format": "csv",
                "fingerprint": csv_fingerprint(absolute),
            }
        )
    evaluation = _evaluation(report_19g_path)
    return attach_canonical_hash(
        {
            "schema": "round19_dataset_card",
            "schema_version": 1,
            "stage": "19h",
            "all_done": False,
            "status": evaluation["status"],
            "datasets": records,
            "evaluation_19g": evaluation,
            "usage_constraints": [
                "Dataset fingerprints describe raw bytes and CSV schema separately.",
                "Stage 19G outcomes must not retroactively change locked model roles.",
            ],
        }
    )


def _parse_dataset(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset must use ID=PATH")
        dataset_id, path = value.split("=", 1)
        if not dataset_id or not path or dataset_id in result:
            raise ValueError(f"Invalid or duplicate dataset specification: {value}")
        result[dataset_id] = Path(path)
    return result


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--report-19g")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    card = build_dataset_card(
        Path(args.project_root),
        _parse_dataset(args.dataset),
        report_19g_path=Path(args.report_19g) if args.report_19g else None,
    )
    if args.output and not args.dry_run:
        write_json(Path(args.output), card)
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
