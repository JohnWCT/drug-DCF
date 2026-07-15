#!/usr/bin/env python3
"""Build a read-only Round 19G preflight sidecar over locked 19F artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.round19_stage19g_lock_adapter import load_verified_lock

REQUIRED_19F_ARTIFACTS = (
    "reports/round19_final_role_proposal.json",
    "reports/round19_stage19f_checkpoint_inventory.csv",
    "reports/round19_stage19f_checkpoint_inventory_summary.json",
    "manifests/stage19f_posthoc_internal_test_manifest.csv",
    "manifests/stage19f_posthoc_internal_test_manifest_dispatch_status.csv",
    "manifests/stage19f_posthoc_internal_test_manifest_dispatch_status.summary.json",
    "manifests/stage19f_posthoc_tcga_manifest.csv",
    "manifests/stage19f_posthoc_tcga_manifest_dispatch_status.csv",
    "manifests/stage19f_posthoc_tcga_manifest_dispatch_status.summary.json",
    "reports/round19_stage19f_posthoc/round19f_15member_ensemble_predictions.csv",
    "reports/round19_stage19f_posthoc/round19f_internal_candidate_metrics.csv",
    "reports/round19_stage19f_posthoc/round19f_tcga_per_target_metrics.csv",
    "reports/round19_stage19f_posthoc/round19f_integrated5_equal_target_mean.csv",
    "reports/round19_stage19f_posthoc/round19f_paired_bootstrap_deltas.csv",
    "reports/round19_stage19f_posthoc/round19f_role_alias_view.csv",
    "reports/round19_stage19f_posthoc/round19f_exploratory_posthoc_summary.json",
    "reports/round19_stage19f_posthoc/round19f_exploratory_posthoc_report.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _artifact_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    binding: dict[str, Any] = {
        "path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                columns = next(reader)
            except StopIteration as exc:
                raise AssertionError(f"CSV artifact is empty: {path}") from exc
            binding.update(
                {
                    "format": "csv",
                    "rows": sum(1 for _ in reader),
                    "schema": columns,
                }
            )
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        binding.update(
            {
                "format": "json",
                "rows": None,
                "schema": {
                    "top_level_type": type(value).__name__,
                    "schema_version": value.get("schema_version")
                    if isinstance(value, dict)
                    else None,
                    "artifact_type": value.get("artifact_type")
                    if isinstance(value, dict)
                    else None,
                },
            }
        )
    else:
        binding.update({"format": "markdown", "rows": None, "schema": None})
    return binding


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def build_gate_manifest(
    *,
    project_root: Path,
    round_root: Path,
    final_lock_path: Path,
    baseline_path: Path,
    config_paths: Iterable[Path],
) -> dict[str, Any]:
    """Create an attestation sidecar; never writes or rewrites the final lock."""
    project_root = Path(project_root)
    final_lock_path = Path(final_lock_path)
    lock_hash_before = sha256_file(final_lock_path)
    lock = load_verified_lock(final_lock_path, project_root)
    baseline = _load_object(Path(baseline_path))
    if baseline.get("artifact_type") != "round19_stage19g_local_baseline":
        raise AssertionError("19G gate requires local baseline metadata")
    if baseline.get("tracked_working_tree_clean") is not True:
        raise AssertionError("Tracked working tree is not explicitly clean")
    if baseline.get("remote_sync_required") is not False:
        raise AssertionError("19G baseline must not require remote synchronization")
    if baseline.get("remote_operations_performed") is not False:
        raise AssertionError("Remote operations are forbidden for the 19G gate")

    artifacts = {
        relative: _artifact_binding(Path(round_root) / relative, relative)
        for relative in REQUIRED_19F_ARTIFACTS
    }
    configs = {}
    for config_path in config_paths:
        path = Path(config_path)
        config = _load_object(path)
        expected_lock = config.get("final_role_lock")
        if isinstance(expected_lock, dict):
            if expected_lock.get("file_sha256") != lock_hash_before:
                raise AssertionError(f"Config final lock file hash mismatch: {path}")
            if (
                expected_lock.get("payload_sha256")
                != lock["hashes"]["lock_payload_sha256"]
            ):
                raise AssertionError(f"Config final lock payload hash mismatch: {path}")
            if (
                expected_lock.get("schema_version") != 1
                or expected_lock.get("immutable") is not True
            ):
                raise AssertionError(f"Config does not require immutable schema-v1 lock: {path}")
        configs[str(path.relative_to(project_root))] = {
            "sha256": sha256_file(path),
            "schema_version": config.get("schema_version"),
            "artifact_type": config.get("artifact_type"),
        }

    if lock_hash_before != sha256_file(final_lock_path):
        raise AssertionError("Final role lock changed during 19G gate construction")
    payload = {
        "artifact_type": "round19_stage19g_preflight_gate_manifest",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "PREFLIGHT_ONLY",
        "formal_stage19g_experiment_lock_created": False,
        "formal_inference_started": False,
        "final_lock_attestation": {
            "status": "LOCKED",
            "compatibility_only": True,
            "authoritative_lock_field": "immutable",
            "immutable": True,
            "lock_path": str(final_lock_path.relative_to(project_root)),
            "lock_file_sha256": lock_hash_before,
            "lock_payload_sha256": lock["hashes"]["lock_payload_sha256"],
        },
        "local_baseline": {
            "path": str(Path(baseline_path).relative_to(project_root)),
            "sha256": sha256_file(Path(baseline_path)),
            "snapshot": baseline,
        },
        "stage19f_artifacts": artifacts,
        "configs": configs,
        "blockers": [
            "case selection manifest is not finalized",
            "formal Stage 19G experiment lock requires a later local commit",
        ],
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    project_root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build Round 19G preflight gate sidecar")
    parser.add_argument("--project-root", default=str(project_root_default))
    parser.add_argument(
        "--round-root",
        default="result/optimization_runs/round19_factorial",
    )
    parser.add_argument(
        "--final-lock",
        default=(
            "result/optimization_runs/round19_factorial/"
            "reports/round19_final_role_lock.json"
        ),
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "config/round19_stage19g_interpretability.json",
            "config/round19_stage19g_case_selection.json",
            "config/round19_stage19h_finalize.json",
        ],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    payload = build_gate_manifest(
        project_root=project_root,
        round_root=rooted(args.round_root),
        final_lock_path=rooted(args.final_lock),
        baseline_path=rooted(args.baseline),
        config_paths=[rooted(value) for value in args.configs],
    )
    output = rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
