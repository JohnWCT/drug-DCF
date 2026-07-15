from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.round19_stage19f_final_lock import REQUIRED_ROLES, canonical_sha256
from tools.round19_stage19g_gate_manifest import (
    REQUIRED_19F_ARTIFACTS,
    build_gate_manifest,
)
from tools.round19_stage19g_local_baseline import (
    build_local_baseline,
    validate_host_snapshot,
)
from tools.round19_stage19g_lock_adapter import load_verified_lock, route_locked


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock_fixture(project_root: Path) -> Path:
    sources = [f"F{index}_candidate" for index in range(6)]
    inventory = []
    for source in sources:
        for seed in (52, 62, 72):
            for fold in range(5):
                member = f"seed{seed}_fold{fold}"
                checkpoint = (
                    project_root
                    / "checkpoints"
                    / f"{source}__seed{seed}__fold{fold}"
                    / "checkpoint.pt"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"{source}:{member}".encode())
                inventory.append(
                    {
                        "source_candidate_id": source,
                        "member_id": member,
                        "checkpoint_path": str(checkpoint.relative_to(project_root)),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "checkpoint_size_bytes": checkpoint.stat().st_size,
                    }
                )
    role_sources = {
        "historical_anchor": "F0",
        "source_performance_champion": "F2",
        "parsimonious_context_model": "F1",
        "cancer_shift_specialist": "F1",
        "chemical_shift_specialist": "F3",
        "source_only_domain_candidate": "F4",
        "efficient_model": "F5",
        "general_recommended_model": "F3",
    }
    lock = {
        "lock_type": "round19_final_role_lock",
        "schema_version": 1,
        "immutable": True,
        "roles": {
            role: {
                "candidate_id": f"E{index}",
                "source_candidate_id": role_sources[role],
            }
            for index, role in enumerate(sorted(REQUIRED_ROLES))
        },
        "single_champion": None,
        "selection_used_internal": False,
        "selection_used_tcga": False,
        "role_immutability": {
            "internal_test_may_change_roles": False,
            "tcga_may_change_roles": False,
            "routing_may_override_locked_roles": False,
        },
        "hashes": {"checkpoint_inventory": inventory},
    }
    lock["hashes"]["lock_payload_sha256"] = canonical_sha256(lock)
    path = project_root / "reports" / "round19_final_role_lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def _write_artifacts(round_root: Path) -> None:
    for relative in REQUIRED_19F_ARTIFACTS:
        path = round_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "value"])
                writer.writerow(["one", "1"])
        elif path.suffix == ".json":
            path.write_text(
                json.dumps({"schema_version": 1, "artifact_type": path.stem}),
                encoding="utf-8",
            )
        else:
            path.write_text("# report\n", encoding="utf-8")


def test_lock_adapter_routes_and_normalizes_short_source_ids(tmp_path):
    lock_path = _lock_fixture(tmp_path)
    before = _sha256(lock_path)
    decision = route_locked(
        lock_path, "unseen_scaffold", project_root=tmp_path
    )
    assert decision["selected_role"] == "chemical_shift_specialist"
    assert decision["source_candidate_id"] == "F3_candidate"
    assert decision["lock_file_sha256"] == before
    assert _sha256(lock_path) == before


def test_lock_adapter_rejects_proposal_object_and_argument(tmp_path):
    lock_path = _lock_fixture(tmp_path)
    with pytest.raises(TypeError, match="proposal_roles is forbidden"):
        route_locked(
            lock_path,
            "source_like",
            project_root=tmp_path,
            proposal_roles={"source_performance_champion": "F2"},
        )
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps({"lock_type": "round19_final_role_proposal"}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="Proposal roles"):
        load_verified_lock(proposal, tmp_path)


def test_host_snapshot_is_explicit_and_unknown_never_means_clean(tmp_path):
    snapshot = {
        "local_head": "a" * 40,
        "branch": "main",
        "tracked_working_tree_clean": True,
        "untracked_present": True,
    }
    baseline = build_local_baseline(tmp_path, host_snapshot=snapshot)
    assert baseline["snapshot_source"] == "explicit_host_snapshot"
    assert baseline["remote_sync_required"] is False
    assert baseline["remote_operations_performed"] is False
    assert baseline["untracked_present"] is True
    snapshot["tracked_working_tree_clean"] = "UNKNOWN"
    with pytest.raises(TypeError, match="explicit boolean"):
        validate_host_snapshot(snapshot)


def test_gate_binds_19f_artifacts_and_preserves_lock(tmp_path):
    project_root = tmp_path
    round_root = project_root / "result" / "round19"
    lock_path = _lock_fixture(project_root)
    _write_artifacts(round_root)
    baseline = build_local_baseline(
        project_root,
        host_snapshot={
            "local_head": "b" * 40,
            "branch": "main",
            "tracked_working_tree_clean": True,
            "untracked_present": True,
        },
    )
    baseline_path = project_root / "metadata" / "baseline.json"
    baseline_path.parent.mkdir()
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    configs = []
    for name in ("interpretability", "case_selection", "finalize"):
        path = project_root / "config" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": 1, "artifact_type": name}),
            encoding="utf-8",
        )
        configs.append(path)

    before = _sha256(lock_path)
    gate = build_gate_manifest(
        project_root=project_root,
        round_root=round_root,
        final_lock_path=lock_path,
        baseline_path=baseline_path,
        config_paths=configs,
    )
    assert gate["final_lock_attestation"]["status"] == "LOCKED"
    assert gate["final_lock_attestation"]["authoritative_lock_field"] == "immutable"
    assert gate["formal_stage19g_experiment_lock_created"] is False
    assert gate["formal_inference_started"] is False
    assert len(gate["stage19f_artifacts"]) == len(REQUIRED_19F_ARTIFACTS)
    assert gate["stage19f_artifacts"][
        "reports/round19_stage19f_posthoc/round19f_15member_ensemble_predictions.csv"
    ]["rows"] == 1
    assert _sha256(lock_path) == before
