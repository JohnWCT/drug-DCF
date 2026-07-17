from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.round20_e3_resolver import E3ResolutionError, resolve_e3

ROOT = Path(__file__).resolve().parents[1]


def test_resolve_e3_from_artifacts() -> None:
    resolved = resolve_e3(ROOT, allow_approved_reconstruction=False)
    assert resolved.reconstructed is False
    assert resolved.is_original_e3 is True
    assert resolved.source_candidate_id == "F3_best_pooled_o2"
    assert resolved.public_alias == "E3"
    assert resolved.architecture_family == "pooled_mlp"
    assert resolved.predictor_id == "P0"
    assert resolved.omics_id == "O2"
    assert resolved.drug_encoder_id == "D0"
    assert resolved.drug_encoder_training_mode == "end_to_end_finetune"
    assert resolved.pooling == "global_max"
    assert resolved.context_dim == 16
    assert resolved.omics_dim == 80
    assert resolved.graph_dim == 32
    assert resolved.node_hidden_dim == 32
    assert resolved.hidden_dims == (128,)
    assert len(resolved.checkpoint_paths) == 15
    assert resolved.evidence["cross_source_agreement"] is True


def test_resolve_e3_fails_when_inventory_empty(tmp_path: Path) -> None:
    lock_src = (
        ROOT
        / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
    )
    if not lock_src.is_file():
        pytest.skip("role lock missing")
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "config").mkdir(parents=True)
    (repo / "result/optimization_runs/round19_factorial/reports").mkdir(parents=True)

    lock = json.loads(lock_src.read_text(encoding="utf-8"))
    lock["hashes"]["checkpoint_inventory"] = []
    (repo / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    (repo / "reports/round19_deployment_policy.json").write_text(
        (ROOT / "reports/round19_deployment_policy.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "config/round19_factorial_settings.json").write_text(
        (ROOT / "config/round19_factorial_settings.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(E3ResolutionError, match="checkpoint inventory"):
        resolve_e3(repo, allow_approved_reconstruction=False)


def test_approved_reconstruction_marked_not_original(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "config").mkdir(parents=True)
    (repo / "result/optimization_runs/round19_factorial/reports").mkdir(parents=True)
    lock = json.loads(
        (
            ROOT
            / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
        ).read_text(encoding="utf-8")
    )
    lock["hashes"]["checkpoint_inventory"] = [
        {
            "checkpoint_path": "missing/F3_best_pooled_o2__seed52__fold0/checkpoint.pt",
            "source_candidate_id": "F3_best_pooled_o2",
            "member_id": "seed52_fold0",
        }
    ]
    (repo / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    (repo / "reports/round19_deployment_policy.json").write_text(
        (ROOT / "reports/round19_deployment_policy.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "config/round19_factorial_settings.json").write_text(
        (ROOT / "config/round19_factorial_settings.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    resolved = resolve_e3(repo, allow_approved_reconstruction=True)
    assert resolved.reconstructed is True
    assert resolved.is_original_e3 is False
    assert resolved.baseline_label == "approved_reconstructed_pooled_mlp"
    assert resolved.evidence["is_original_e3"] is False
