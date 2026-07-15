import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import round19_reproducibility_audit as audit
from tools.round19_artifact_manifest import build_artifact_manifest
from tools.round19_dataset_card_builder import build_dataset_card
from tools.round19_model_card_builder import build_model_card


def _make_lock(root: Path, count: int = 90) -> Path:
    inventory = []
    for index in range(count):
        path = root / "checkpoints" / f"checkpoint_{index:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"checkpoint-{index}".encode())
        inventory.append({"checkpoint_path": path.relative_to(root).as_posix(), "checkpoint_sha256": audit.sha256_file(path)})
    lock = {"lock_type": "round19_final_role_lock", "immutable": True, "single_champion": None, "roles": {"historical_anchor": {"source_candidate_id": "candidate"}}, "hashes": {"lock_payload_sha256": "synthetic", "checkpoint_inventory": inventory}}
    path = root / "reports" / "round19_final_role_lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_hashes_csv_and_canonical_attestation_are_deterministic(tmp_path: Path):
    data = tmp_path / "dataset.csv"
    data.write_bytes(b"a,b\n1,2\n3,4\n")
    expected = hashlib.sha256(data.read_bytes()).hexdigest()
    assert audit.sha256_file(data, chunk_size=2) == expected
    fingerprint = audit.csv_fingerprint(data)
    assert fingerprint["raw_sha256"] == expected
    assert fingerprint["row_count"] == 2
    assert fingerprint["schema"]["columns"] == ["a", "b"]
    first = {"value": 1, "attestation": {"created_at_utc": "one"}}
    second = {"attestation": {"created_at_utc": "two"}, "value": 1}
    assert audit.canonical_json_hash(audit.canonical_payload(first)) == audit.canonical_json_hash(audit.canonical_payload(second))


def test_symlink_audit_records_literal_resolved_and_flags(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"weights")
    link = root / "absolute_checkpoint.pt"
    link.symlink_to(outside)
    record = audit.audit_symlink(link, root)
    assert record["literal_target"] == str(outside)
    assert record["resolved_source"] == str(outside)
    assert record["content_sha256"] == audit.sha256_file(outside)
    assert record["absolute_target"] is True
    assert record["outside_project"] is True
    assert record["broken"] is False
    outside.unlink()
    assert audit.audit_symlink(link, root)["broken"] is True


def test_tree_manifest_is_stable_and_does_not_follow_directory_links(tmp_path: Path):
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x.txt").write_text("x", encoding="utf-8")
    (root / "alias").symlink_to(root / "data", target_is_directory=True)
    first = audit.tree_manifest(root, [root / "data", root / "alias"])
    second = audit.tree_manifest(root, [root / "alias", root / "data"])
    assert first == second
    assert [row["path"] for row in first] == ["alias", "data/x.txt"]


def test_git_unknown_is_a_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(audit, "_run_diagnostic", lambda command, cwd=None: {"available": False, "error": "missing"})
    state = audit.collect_git_state(tmp_path)
    assert state["commit"] == "UNKNOWN"
    assert state["valid"] is False
    assert state["remote_sync_required"] is False
    assert state["failure_reasons"] == ["git_commit_unknown"]


def test_audit_requires_90_and_repeated_hash_is_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lock = _make_lock(tmp_path)
    monkeypatch.setattr(audit, "collect_git_state", lambda root: {"commit": "abc", "branch": "main", "working_tree_dirty": False, "status_sha256": "status", "remote_sync_required": False, "valid": True, "failure_reasons": []})
    monkeypatch.setattr(audit, "collect_environment_metadata", lambda: {"python": "fixed"})
    first = audit.build_reproducibility_audit(tmp_path, lock)
    second = audit.build_reproducibility_audit(tmp_path, lock)
    assert first["status"] == "pass"
    assert first["final_role_lock"]["checkpoint_count"] == 90
    assert first["canonical_sha256"] == second["canonical_sha256"]
    assert first["all_done"] is False
    incomplete_root = tmp_path / "incomplete"
    incomplete = _make_lock(incomplete_root, count=1)
    with pytest.raises(AssertionError, match="90"):
        audit.build_reproducibility_audit(incomplete_root, incomplete)


def test_retention_forces_lock_manifest_reachable_and_90_checkpoints_keep(tmp_path: Path):
    lock = _make_lock(tmp_path)
    linked = tmp_path / "reports" / "metrics.csv"
    linked.write_text("metric,value\nauc,0.5\n", encoding="utf-8")
    manifest = tmp_path / "reports" / "inputs_manifest.json"
    manifest.write_text(json.dumps({"metrics_path": "reports/metrics.csv"}), encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}_external.pt"
    outside.write_bytes(b"external")
    absolute_link = tmp_path / "reports" / "container_checkpoint.pt"
    absolute_link.symlink_to(outside)
    try:
        result, sidecar = build_artifact_manifest(tmp_path, lock, [tmp_path / "reports"], manifest_seeds=[manifest])
        by_path = {row["path"]: row for row in result["artifacts"]}
        assert by_path["reports/round19_final_role_lock.json"]["retention"] == "KEEP"
        assert by_path["reports/metrics.csv"]["retention"] == "KEEP"
        assert all(by_path[f"checkpoints/checkpoint_{index:03d}.pt"]["retention"] == "KEEP" for index in range(90))
        assert result["locked_checkpoint_count"] == 90
        assert result["checkpoint_copy_performed"] is False
        assert all(row["operation"] == "plan_only_no_copy" for row in result["archive_plan"])
        assert sidecar["source_links_modified"] is False
        assert sidecar["mappings"][0]["archive_path"] == "reports/container_checkpoint.pt"
        assert os.readlink(absolute_link) == str(outside)
    finally:
        outside.unlink(missing_ok=True)


def test_cards_await_for_19g_and_only_use_explicit_verdict(tmp_path: Path):
    lock = _make_lock(tmp_path)
    dataset = tmp_path / "data.csv"
    dataset.write_text("drug,response\nA,1\n", encoding="utf-8")
    model = build_model_card(lock)
    data = build_dataset_card(tmp_path, {"primary": Path("data.csv")})
    assert model["status"] == data["status"] == "awaiting_19g"
    assert model["evaluation_19g"]["verdict"] is None
    assert data["evaluation_19g"]["verdict"] is None
    report = tmp_path / "19g.json"
    report.write_text(json.dumps({"stage": "19g", "verdict": "qualified_pass"}))
    assert build_model_card(lock, report_19g_path=report)["evaluation_19g"]["verdict"] == "qualified_pass"
    report.write_text(json.dumps({"stage": "19g"}))
    with pytest.raises(ValueError, match="explicit"):
        build_dataset_card(tmp_path, {"primary": Path("data.csv")}, report_19g_path=report)


def test_finalize_script_defaults_safe_and_never_declares_all_done():
    script = (Path(__file__).parents[1] / "tools" / "run_round19_stage19h_finalize.sh").read_text(encoding="utf-8")
    assert "DRY_RUN=1" in script
    assert "REQUIRE_COMPLETE=1" in script
    assert "git push" not in script.lower()
    assert "rm " not in script
    assert "ALL_DONE marker occurred" in script
    assert "touch" not in script
