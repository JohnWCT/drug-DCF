#!/usr/bin/env python3
"""Build post-hoc inference manifests from the immutable Round 19F lock only.

This module intentionally does not import selectors or read Stage 19D/19E
manifests, rankings, metrics, proposals, or policies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REQUIRED_ROLES = {
    "historical_anchor",
    "source_performance_champion",
    "parsimonious_context_model",
    "cancer_shift_specialist",
    "chemical_shift_specialist",
    "source_only_domain_candidate",
    "efficient_model",
    "general_recommended_model",
}
REQUIRED_SEEDS = (52, 62, 72)
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_MEMBERS = {
    f"seed{seed}_fold{fold}" for seed in REQUIRED_SEEDS for fold in REQUIRED_FOLDS
}
TCGA_TARGETS: Tuple[Tuple[str, str], ...] = (
    (
        "gdsc_intersect13",
        "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv",
    ),
    (
        "tcga_only3",
        "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv",
    ),
    ("dapl", "data/TCGA/TCGA_drug_response_from_DAPL.csv"),
    (
        "aacdr_tcga_only",
        "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv",
    ),
    (
        "aacdr_gdsc_intersect",
        "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv",
    ),
)
MEMBER_RE = re.compile(r"^seed(?P<seed>\d+)_fold(?P<fold>\d+)$")
IDENTITY_KEYS = ("drug_id", "predictor_id", "omics_id")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_checkpoint_identity(path: Path) -> Dict[str, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint identity verification requires PyTorch; run in the DAPL Docker image"
        ) from exc
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Cannot deserialize checkpoint: {path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise AssertionError(f"Checkpoint payload must be a mapping: {path}")
    missing = [key for key in IDENTITY_KEYS if not checkpoint.get(key)]
    if missing:
        raise AssertionError(f"Checkpoint missing identity {missing}: {path}")
    return {key: str(checkpoint[key]) for key in IDENTITY_KEYS}


def load_and_verify_final_lock(lock_path: Path, project_root: Path) -> Dict[str, Any]:
    """Verify lock payload, role policy, checkpoint hashes, identity, and members."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise TypeError("Final lock must be a JSON object")
    if lock.get("lock_type") != "round19_final_role_lock":
        raise AssertionError("Not a Round 19 final role lock")
    if lock.get("schema_version") != 1 or lock.get("immutable") is not True:
        raise AssertionError("Final role lock must be schema v1 and immutable")
    if lock.get("single_champion") is not None:
        raise AssertionError("Post-hoc inference must not introduce a single champion")
    if lock.get("selection_used_internal") is not False:
        raise AssertionError("Lock says internal outcomes participated in selection")
    if lock.get("selection_used_tcga") is not False:
        raise AssertionError("Lock says TCGA outcomes participated in selection")
    if set(lock.get("roles", {})) != REQUIRED_ROLES:
        raise AssertionError("Final role lock has incomplete or unexpected roles")

    immutability = lock.get("role_immutability", {})
    forbidden = (
        "internal_test_may_change_roles",
        "tcga_may_change_roles",
        "routing_may_override_locked_roles",
    )
    if any(immutability.get(key) is not False for key in forbidden):
        raise AssertionError("Final role lock does not forbid post-hoc role changes")

    hashes = lock.get("hashes")
    if not isinstance(hashes, dict):
        raise AssertionError("Final lock has no hashes object")
    expected_payload_hash = hashes.get("lock_payload_sha256")
    payload = json.loads(json.dumps(lock, allow_nan=False))
    payload["hashes"].pop("lock_payload_sha256", None)
    if expected_payload_hash != canonical_sha256(payload):
        raise AssertionError("Final role lock payload hash mismatch")

    inventory = hashes.get("checkpoint_inventory")
    if not isinstance(inventory, list) or len(inventory) != 90:
        raise AssertionError("Final lock must pin exactly 90 checkpoints")

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    seen_paths = set()
    for item in inventory:
        if not isinstance(item, Mapping):
            raise TypeError("Checkpoint inventory entries must be objects")
        required = {
            "source_candidate_id",
            "member_id",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
        }
        missing = required - set(item)
        if missing:
            raise KeyError(f"Checkpoint lock entry missing fields: {sorted(missing)}")
        source = str(item["source_candidate_id"])
        member = str(item["member_id"])
        match = MEMBER_RE.fullmatch(member)
        if not match or member not in REQUIRED_MEMBERS:
            raise AssertionError(f"Unexpected checkpoint member identity: {source}/{member}")
        path_text = str(item["checkpoint_path"])
        if path_text in seen_paths:
            raise AssertionError(f"Checkpoint path appears more than once: {path_text}")
        seen_paths.add(path_text)
        path = _resolve_path(path_text, project_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["checkpoint_size_bytes"]):
            raise AssertionError(f"Checkpoint size mismatch: {path}")
        if sha256_file(path) != item["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch: {path}")
        expected_dir = f"{source}__{member.replace('_fold', '__fold')}"
        if path.name != "checkpoint.pt" or path.parent.name != expected_dir:
            raise AssertionError(
                f"Checkpoint path identity mismatch for {source}/{member}: {path}"
            )
        grouped[source].append(item)

    if len(grouped) != 6:
        raise AssertionError(f"Expected 6 source candidates, got {len(grouped)}")
    identities: Dict[str, Dict[str, str]] = {}
    for source, items in grouped.items():
        members = [str(item["member_id"]) for item in items]
        if len(members) != 15 or set(members) != REQUIRED_MEMBERS:
            raise AssertionError(f"{source} does not have the complete 15-member ensemble")
        if len(set(members)) != 15:
            raise AssertionError(f"{source} has duplicate member identities")
        member_identities = {
            tuple(
                _load_checkpoint_identity(
                    _resolve_path(str(item["checkpoint_path"]), project_root)
                )[key]
                for key in IDENTITY_KEYS
            )
            for item in items
        }
        if len(member_identities) != 1:
            raise AssertionError(f"{source} checkpoint model identities are inconsistent")
        identities[source] = dict(zip(IDENTITY_KEYS, next(iter(member_identities))))

    lock["_verified_checkpoint_identities"] = identities
    return lock


def _role_aliases_by_source(
    roles: Mapping[str, Mapping[str, Any]], sources: Iterable[str]
) -> Dict[str, List[str]]:
    available = set(sources)
    aliases: Dict[str, List[str]] = defaultdict(list)
    for role, record in roles.items():
        requested = str(record.get("source_candidate_id") or "")
        if not requested:
            raise AssertionError(f"Role has no source_candidate_id: {role}")
        matches = sorted(
            source
            for source in available
            if source == requested
            or source.startswith(requested + "_")
            or requested.startswith(source + "_")
        )
        if len(matches) != 1:
            raise AssertionError(
                f"Role {role} source {requested!r} does not uniquely resolve: {matches}"
            )
        aliases[matches[0]].append(role)
    if set(aliases) != available:
        raise AssertionError("Every locked source candidate must have at least one role alias")
    return {source: sorted(names) for source, names in aliases.items()}


def _base_rows(lock: Mapping[str, Any]) -> List[Dict[str, Any]]:
    inventory = lock["hashes"]["checkpoint_inventory"]
    identities = lock["_verified_checkpoint_identities"]
    aliases = _role_aliases_by_source(lock["roles"], identities)
    rows = []
    for item in sorted(
        inventory, key=lambda row: (row["source_candidate_id"], row["member_id"])
    ):
        source = str(item["source_candidate_id"])
        match = MEMBER_RE.fullmatch(str(item["member_id"]))
        assert match is not None
        rows.append(
            {
                "lock_payload_sha256": lock["hashes"]["lock_payload_sha256"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "checkpoint_path": item["checkpoint_path"],
                "checkpoint_size_bytes": int(item["checkpoint_size_bytes"]),
                "source_candidate_id": source,
                "role_aliases": ",".join(aliases[source]),
                "member_id": item["member_id"],
                "split_seed": int(match.group("seed")),
                "fold_id": int(match.group("fold")),
                **identities[source],
            }
        )
    return rows


def build_posthoc_manifests(
    *,
    final_lock_path: Path,
    project_root: Path,
    output_root: Path,
    internal_output: Path,
    tcga_output: Path,
    verify_target_paths: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lock = load_and_verify_final_lock(final_lock_path, project_root)
    base_rows = _base_rows(lock)
    internal_rows = []
    tcga_rows = []
    for row in base_rows:
        source = row["source_candidate_id"]
        member = row["member_id"]
        internal_rows.append(
            {
                "job_id": f"internal__{source}__{member}",
                **row,
                "mode": "infer_internal_test",
                "target": "internal_test",
                "target_path": "",
                "result_dir": str(output_root / "internal_test" / source / member),
            }
        )
        for target, target_path in TCGA_TARGETS:
            if verify_target_paths and not _resolve_path(target_path, project_root).is_file():
                raise FileNotFoundError(_resolve_path(target_path, project_root))
            tcga_rows.append(
                {
                    "job_id": f"tcga__{target}__{source}__{member}",
                    **row,
                    "mode": "infer_tcga",
                    "target": target,
                    "target_path": target_path,
                    "result_dir": str(output_root / "tcga" / target / source / member),
                }
            )
    if len(internal_rows) != 90 or len(tcga_rows) != 450:
        raise AssertionError(
            f"Manifest cardinality mismatch: internal={len(internal_rows)} tcga={len(tcga_rows)}"
        )
    _write_csv(internal_output, internal_rows)
    _write_csv(tcga_output, tcga_rows)
    return internal_rows, tcga_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build Round 19F post-hoc manifests from the final lock only"
    )
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--project-root", default=str(project_root_default))
    parser.add_argument(
        "--output-root",
        default="result/optimization_runs/round19_factorial/stage19f_posthoc",
    )
    parser.add_argument(
        "--internal-output",
        default=(
            "result/optimization_runs/round19_factorial/manifests/"
            "stage19f_posthoc_internal_test_manifest.csv"
        ),
    )
    parser.add_argument(
        "--tcga-output",
        default=(
            "result/optimization_runs/round19_factorial/manifests/"
            "stage19f_posthoc_tcga_manifest.csv"
        ),
    )
    parser.add_argument(
        "--skip-target-path-check",
        action="store_true",
        help="Synthetic-test escape hatch; checkpoint verification is never skipped",
    )
    args = parser.parse_args()
    internal, tcga = build_posthoc_manifests(
        final_lock_path=Path(args.final_lock),
        project_root=Path(args.project_root),
        output_root=Path(args.output_root),
        internal_output=Path(args.internal_output),
        tcga_output=Path(args.tcga_output),
        verify_target_paths=not args.skip_target_path_check,
    )
    print(
        json.dumps(
            {
                "final_lock": args.final_lock,
                "lock_payload_sha256": internal[0]["lock_payload_sha256"],
                "internal_jobs": len(internal),
                "tcga_jobs": len(tcga),
                "unique_source_candidates": len(
                    {row["source_candidate_id"] for row in internal}
                ),
                "internal_output": args.internal_output,
                "tcga_output": args.tcga_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
