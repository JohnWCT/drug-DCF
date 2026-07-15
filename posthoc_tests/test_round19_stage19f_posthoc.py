import csv
import json
from pathlib import Path

import pytest
import torch

from tools import round19_stage19f_inference_dispatch as dispatch
from tools import round19_stage19f_posthoc_manifest as posthoc


SOURCES = (
    "F0_historical_anchor",
    "F1_primary_o2",
    "F2_full_omics_o3",
    "F3_best_pooled_o2",
    "F4_source_only_o4",
    "F5_maccs_efficient",
)
ROLES = {
    "historical_anchor": {"source_candidate_id": "F0_historical_anchor"},
    "source_performance_champion": {"source_candidate_id": "F2"},
    "parsimonious_context_model": {"source_candidate_id": "F1"},
    "cancer_shift_specialist": {"source_candidate_id": "F1_primary_o2"},
    "chemical_shift_specialist": {"source_candidate_id": "F3_best_pooled_o2"},
    "source_only_domain_candidate": {"source_candidate_id": "F4_source_only_o4"},
    "efficient_model": {"source_candidate_id": "F5_maccs_efficient"},
    "general_recommended_model": {"source_candidate_id": "F3_best_pooled_o2"},
}


def _make_lock(tmp_path: Path) -> Path:
    inventory = []
    for source_index, source in enumerate(SOURCES):
        identity = {
            "drug_id": f"D{source_index}",
            "predictor_id": f"P{source_index}",
            "omics_id": f"O{source_index}",
        }
        for seed in posthoc.REQUIRED_SEEDS:
            for fold in posthoc.REQUIRED_FOLDS:
                member = f"seed{seed}_fold{fold}"
                checkpoint_dir = f"{source}__{member.replace('_fold', '__fold')}"
                checkpoint = tmp_path / "checkpoints" / checkpoint_dir / "checkpoint.pt"
                checkpoint.parent.mkdir(parents=True)
                torch.save({**identity, "head": {"weight": torch.tensor([source_index])}}, checkpoint)
                inventory.append(
                    {
                        "source_candidate_id": source,
                        "member_id": member,
                        "checkpoint_path": str(checkpoint.relative_to(tmp_path)),
                        "checkpoint_sha256": posthoc.sha256_file(checkpoint),
                        "checkpoint_size_bytes": checkpoint.stat().st_size,
                    }
                )
    lock = {
        "lock_type": "round19_final_role_lock",
        "schema_version": 1,
        "immutable": True,
        "roles": ROLES,
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
    lock["hashes"]["lock_payload_sha256"] = posthoc.canonical_sha256(lock)
    path = tmp_path / "final_lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_final_lock_only_builder_writes_90_and_450_jobs(tmp_path):
    lock_path = _make_lock(tmp_path)
    internal_path = tmp_path / "internal.csv"
    tcga_path = tmp_path / "tcga.csv"
    internal, tcga = posthoc.build_posthoc_manifests(
        final_lock_path=lock_path,
        project_root=tmp_path,
        output_root=tmp_path / "results",
        internal_output=internal_path,
        tcga_output=tcga_path,
        verify_target_paths=False,
    )

    assert len(internal) == 90
    assert len(tcga) == 450
    assert {row["mode"] for row in internal} == {"infer_internal_test"}
    assert {row["mode"] for row in tcga} == {"infer_tcga"}
    assert {row["target"] for row in tcga} == {target for target, _ in posthoc.TCGA_TARGETS}
    assert all(row["lock_payload_sha256"] for row in internal + tcga)
    f1 = [row for row in internal if row["source_candidate_id"] == "F1_primary_o2"]
    assert {
        "cancer_shift_specialist",
        "parsimonious_context_model",
    } == set(f1[0]["role_aliases"].split(","))
    assert len({(row["source_candidate_id"], row["member_id"]) for row in internal}) == 90

    with internal_path.open(newline="", encoding="utf-8") as handle:
        header = set(csv.DictReader(handle).fieldnames)
    assert {
        "lock_payload_sha256",
        "checkpoint_sha256",
        "role_aliases",
        "mode",
        "target",
        "result_dir",
    } <= header


def test_builder_rejects_checkpoint_hash_change(tmp_path):
    lock_path = _make_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / lock["hashes"]["checkpoint_inventory"][0]["checkpoint_path"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(AssertionError, match="size mismatch|hash mismatch"):
        posthoc.load_and_verify_final_lock(lock_path, tmp_path)


def test_builder_rejects_incomplete_ensemble(tmp_path):
    lock_path = _make_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["hashes"]["checkpoint_inventory"].pop()
    lock["hashes"].pop("lock_payload_sha256")
    lock["hashes"]["lock_payload_sha256"] = posthoc.canonical_sha256(lock)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(AssertionError, match="exactly 90"):
        posthoc.load_and_verify_final_lock(lock_path, tmp_path)


def test_oom_retry_preserves_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    checkpoint_hash = posthoc.sha256_file(checkpoint)
    fake_pipeline = tmp_path / "fake_pipeline.py"
    fake_pipeline.write_text(
        "import sys\n"
        "mb = int(sys.argv[sys.argv.index('--micro-batch-size') + 1])\n"
        "raise SystemExit(42 if mb > 32 else 0)\n",
        encoding="utf-8",
    )
    job = {
        "job_id": "synthetic",
        "mode": "infer_internal_test",
        "target": "internal_test",
        "target_path": "",
        "result_dir": str(tmp_path / "result"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "lock_payload_sha256": "a" * 64,
        "source_candidate_id": "F0",
        "drug_id": "D0",
        "predictor_id": "P0",
        "omics_id": "O1",
        "fold_id": "0",
        "split_seed": "52",
    }
    status = dispatch.run_job_with_oom_retry(
        job=job,
        pipeline=str(fake_pipeline),
        python_exe="python3",
        cuda_device="",
        micro_batch_candidates=(64, 32),
        max_retries=1,
        settings="unused.json",
        internal_test_path="unused.csv",
        project_root=tmp_path,
    )
    assert status["status"] == "done"
    assert status["oom_batch_history"] == [64]
    assert checkpoint.is_file()
    assert posthoc.sha256_file(checkpoint) == checkpoint_hash


def test_future_pipeline_command_and_vram_packing(monkeypatch):
    job = {
        "mode": "infer_tcga",
        "result_dir": "out",
        "checkpoint_path": "checkpoint.pt",
        "source_candidate_id": "F0",
        "target": "dapl",
        "target_path": "target.csv",
        "drug_id": "D0",
        "predictor_id": "P0",
        "omics_id": "O1",
        "fold_id": "4",
        "split_seed": "72",
    }
    command = dispatch.build_inference_command(
        job=job,
        pipeline="step1_finetune_latent_pipeline_round19.py",
        python_exe="python3",
        micro_batch=128,
        settings="settings.json",
        internal_test_path="internal.csv",
    )
    assert command[command.index("--mode") + 1] == "infer_tcga"
    assert command[command.index("--target-path") + 1] == "target.csv"
    assert command[command.index("--split-seed") + 1] == "72"
    assert command[command.index("--source-candidate-id") + 1] == "F0"

    monkeypatch.setattr(dispatch, "_nvidia_gpu_memory", lambda: [("0", 24000, 24000)])
    slots, plan = dispatch.build_gpu_slots(
        target_vram_fraction=0.90, estimated_job_mb=3500, reserve_mb=512
    )
    assert len(slots) == 6
    assert plan[0]["packing_budget_mb"] <= int(24000 * 0.90)


def test_builder_has_no_selector_or_ranking_dependency():
    source = Path(posthoc.__file__).read_text(encoding="utf-8")
    assert "role_selector" not in source
    assert "stage19d_manifest" not in source
    assert "stage19e" not in source.lower()
