from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS
from tools.round19_stage19f_final_lock import REQUIRED_ROLES, canonical_sha256
from tools.round19_stage19f_posthoc import CLASSIFICATION, TCGA_TARGETS, analyze_posthoc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    candidates = tuple(f"F{index}_candidate" for index in range(6))
    inventory = []
    for candidate in candidates:
        for member in REQUIRED_MEMBER_IDS:
            checkpoint = tmp_path / "checkpoints" / candidate / f"{member}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{candidate}:{member}".encode())
            inventory.append(
                {
                    "source_candidate_id": candidate,
                    "member_id": member,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                }
            )

    role_candidates = {
        "historical_anchor": candidates[0],
        "source_performance_champion": candidates[1],
        "parsimonious_context_model": candidates[2],
        "cancer_shift_specialist": candidates[3],
        "chemical_shift_specialist": candidates[4],
        "source_only_domain_candidate": candidates[5],
        "efficient_model": candidates[5],
        "general_recommended_model": candidates[4],
    }
    roles = {
        role: {"candidate_id": f"E{index}", "source_candidate_id": role_candidates[role]}
        for index, role in enumerate(REQUIRED_ROLES)
    }
    lock = {
        "lock_type": "round19_final_role_lock",
        "schema_version": 1,
        "immutable": True,
        "roles": roles,
        "single_champion": None,
        "selection_used_internal": False,
        "selection_used_tcga": False,
        "posthoc_classification": CLASSIFICATION,
        "role_immutability": {
            "internal_test_may_change_roles": False,
            "tcga_may_change_roles": False,
            "routing_may_override_locked_roles": False,
        },
        "hashes": {"checkpoint_inventory": inventory},
    }
    lock["hashes"]["lock_payload_sha256"] = canonical_sha256(lock)
    lock_path = tmp_path / "round19_final_role_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    prediction_root = tmp_path / "predictions"
    internal_rows = []
    tcga_rows = []
    hashes = {
        (item["source_candidate_id"], item["member_id"]): item["checkpoint_sha256"]
        for item in inventory
    }
    paths = {
        (item["source_candidate_id"], item["member_id"]): item["checkpoint_path"]
        for item in inventory
    }
    for candidate_index, candidate in enumerate(candidates):
        for target in ("internal_test", *TCGA_TARGETS):
            for row_index in range(10):
                for member_index, member in enumerate(REQUIRED_MEMBER_IDS):
                    row = {
                        "source_candidate_id": candidate,
                        "candidate_id": candidate,
                        "member_id": member,
                        "checkpoint_sha256": hashes[(candidate, member)],
                        "checkpoint_path": paths[(candidate, member)],
                        "target_key": target,
                        "eval_row_id": f"{target}:row{row_index}",
                        "Label": row_index % 2,
                        "ModelID": f"patient{row_index}",
                        "drug_name": "drugA",
                        "probability": (
                            0.15
                            + 0.65 * (row_index % 2)
                            + 0.01 * candidate_index
                            + 0.0001 * member_index
                        ),
                    }
                    (internal_rows if target == "internal_test" else tcga_rows).append(row)
    internal_path = prediction_root / "internal" / "internal_test_predictions.csv"
    tcga_path = prediction_root / "tcga" / "tcga_predictions.csv"
    internal_path.parent.mkdir(parents=True)
    tcga_path.parent.mkdir(parents=True)
    pd.DataFrame(internal_rows).to_csv(internal_path, index=False)
    pd.DataFrame(tcga_rows).to_csv(tcga_path, index=False)
    return lock_path, prediction_root


def test_outputs_exploratory_metrics_and_role_aliases(tmp_path):
    lock_path, prediction_root = _fixture(tmp_path)
    result = analyze_posthoc(lock_path, prediction_root, n_bootstrap=10)
    output = Path(result["summary_path"]).parent
    internal = pd.read_csv(output / "round19f_internal_candidate_metrics.csv")
    tcga = pd.read_csv(output / "round19f_tcga_per_target_metrics.csv")
    integrated = pd.read_csv(output / "round19f_integrated5_equal_target_mean.csv")
    aliases = pd.read_csv(output / "round19f_role_alias_view.csv")
    assert len(internal) == 6
    assert len(tcga) == 30
    assert set(tcga["target_key"]) == set(TCGA_TARGETS)
    assert (integrated["Integrated5_n_targets"] == 5).all()
    assert set(aliases["role_name"]) == REQUIRED_ROLES
    assert set(internal["classification"]) == {CLASSIFICATION}
    assert result["roles_changed"] is False
    payload = Path(result["summary_path"]).read_text()
    assert "winner" not in payload
    assert "prefer_for_lock" not in payload


def test_missing_prediction_file_fails_closed(tmp_path):
    lock_path, prediction_root = _fixture(tmp_path)
    (prediction_root / "tcga" / "tcga_predictions.csv").unlink()
    with pytest.raises(AssertionError, match="target coverage"):
        analyze_posthoc(lock_path, prediction_root, n_bootstrap=2)


def test_requires_all_15_members(tmp_path):
    lock_path, prediction_root = _fixture(tmp_path)
    path = prediction_root / "internal" / "internal_test_predictions.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["member_id"] != REQUIRED_MEMBER_IDS[-1]]
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="exactly 15 members"):
        analyze_posthoc(lock_path, prediction_root, n_bootstrap=2)


def test_identity_drift_fails_closed(tmp_path):
    lock_path, prediction_root = _fixture(tmp_path)
    path = prediction_root / "tcga" / "tcga_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "checkpoint_sha256"] = "0" * 64
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="checkpoint identity drift"):
        analyze_posthoc(lock_path, prediction_root, n_bootstrap=2)


def test_cli_smoke(tmp_path):
    lock_path, prediction_root = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "tools/round19_stage19f_posthoc.py",
            "--final-lock",
            str(lock_path),
            "--prediction-root",
            str(prediction_root),
            "--n-bootstrap",
            "2",
        ],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["classification"] == CLASSIFICATION
