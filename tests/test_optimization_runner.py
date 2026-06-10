import os

import pandas as pd

from tools.optimization_runner import (
    FINETUNE_MANIFEST_COLUMNS,
    ManifestManager,
    build_finetune_manifest,
    run_pretrain_stage,
)


def test_pending_job_selection(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"job_id": "a", "status": "pending", "config_path": "x.json"},
            {"job_id": "b", "status": "success", "config_path": "y.json"},
        ]
    ).to_csv(manifest, index=False)
    mgr = ManifestManager(str(manifest), default_columns=["job_id", "status", "config_path"])
    pending = mgr.pending_jobs()
    assert pending["job_id"].tolist() == ["a"]


def test_success_and_failed_transitions(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"job_id": "a", "status": "pending", "config_path": "missing.json"}]).to_csv(manifest, index=False)
    run_dir = tmp_path / "run"
    run_pretrain_stage(str(manifest), str(run_dir), dry_run=False)
    mgr = ManifestManager(str(manifest), default_columns=["job_id", "status", "config_path", "error_message", "start_time", "end_time", "result_dir"])
    row = mgr.df.iloc[0]
    assert row["status"] == "failed"
    assert "Missing config" in str(row["error_message"])


def test_resume_skips_completed(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"job_id": "a", "status": "success", "config_path": "x.json"},
            {"job_id": "b", "status": "pending", "config_path": "missing.json"},
        ]
    ).to_csv(manifest, index=False)
    cmds = []
    run_pretrain_stage(str(manifest), str(tmp_path / "run"), dry_run=True)
    mgr = ManifestManager(str(manifest), default_columns=["job_id", "status", "config_path"])
    assert mgr.pending_jobs()["job_id"].tolist() == ["b"]


def test_dry_run_prints_commands(tmp_path, capsys):
    config = tmp_path / "cfg.json"
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"job_id": "a", "status": "pending", "config_path": str(config)}]).to_csv(manifest, index=False)
    run_pretrain_stage(str(manifest), str(tmp_path / "run"), dry_run=True)
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "pretrain_VAEwC.py" in captured.out


def test_finetune_manifest_generates_40_jobs(tmp_path):
    top10 = tmp_path / "top10.csv"
    pd.DataFrame({"ID": [f"exp_{i:03d}" for i in range(10)], "result_folder": [f"exp_{i:03d}" for i in range(10)]}).to_csv(top10, index=False)
    manifest_path = build_finetune_manifest(str(top10), str(tmp_path / "run"), force=True)
    df = pd.read_csv(manifest_path)
    assert len(df) == 40
    assert list(df.columns) == FINETUNE_MANIFEST_COLUMNS
