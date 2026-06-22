"""Round 9 pipeline CLI smoke tests (no full pretrain/finetune)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from tests.round9_test_helpers import write_minimal_checkpoint
from tools.build_round9_finetune_select import build_round9_finetune_select
from tools.build_round9_reproduction_manifest import build_reproduction_manifest, _read_baseline_params
from tools.round9_baseline_resolver import main as resolver_main


def _run_cli(module: str, args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m" if False else "", module, *args]
    cmd = [sys.executable, module, *args]
    return subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(module)) if False else None, capture_output=True, text=True)


def test_baseline_resolver_cli(tmp_path):
    exp_dir = write_minimal_checkpoint(str(tmp_path), "exp_048")
    cfg = tmp_path / "baselines.json"
    cfg.write_text(json.dumps({
        "baselines": [{"exp_id": "exp_048", "role": "primary", "required": True, "explicit_path": str(exp_dir)}]
    }))
    outdir = tmp_path / "baselines"
    rc = subprocess.run(
        [sys.executable, "tools/round9_baseline_resolver.py",
         "--baseline-config", str(cfg),
         "--search-root", str(tmp_path),
         "--outdir", str(outdir)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    assert (outdir / "resolved_baselines.csv").exists()


def test_reproduction_manifest_cli(tmp_path):
    exp_dir = write_minimal_checkpoint(str(tmp_path), "exp_048")
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{"exp_id": "exp_048", "role": "primary", "required": True, "resolved": True, "checkpoint_dir": exp_dir}]).to_csv(resolved, index=False)
    baseline_cfg = tmp_path / "round9.json"
    baseline_cfg.write_text(json.dumps({"seeds": [101, 202, 303]}))
    repro = tmp_path / "repro"
    rc = subprocess.run(
        [sys.executable, "tools/build_round9_reproduction_manifest.py",
         "--resolved-baselines", str(resolved),
         "--baseline-config", str(baseline_cfg),
         "--outdir", str(repro),
         "--force"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    assert (repro / "manifests" / "pretrain_sweep_manifest.csv").exists()


def test_read_baseline_params_fallback_run_summary(tmp_path):
    exp_dir = write_minimal_checkpoint(str(tmp_path), "exp_048")
    params_path = os.path.join(exp_dir, "params.json")
    os.remove(params_path)
    with open(os.path.join(exp_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"params": {"latent_size": 64, "encoder_dims": [8, 4], "random_seed": 42}}, f)
    params = _read_baseline_params(exp_dir)
    assert params["latent_size"] == 64


def test_finetune_select_writes_csv(tmp_path):
    repro = tmp_path / "repro"
    exp_dir = write_minimal_checkpoint(str(repro), "exp_001")
    manifest_dir = repro / "manifests"
    manifest_dir.mkdir(parents=True)
    pd.DataFrame([{
        "job_id": "r9_exp_048_seed101",
        "status": "success",
        "result_dir": str(exp_dir),
        "source_exp_id": "exp_048",
        "source_role": "primary",
        "reproduction_seed": 101,
    }]).to_csv(manifest_dir / "pretrain_sweep_manifest.csv", index=False)
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{"exp_id": "exp_048", "resolved": True}]).to_csv(resolved, index=False)
    df = build_round9_finetune_select(str(repro), str(resolved))
    out = tmp_path / "selection"
    out.mkdir()
    df.to_csv(out / "model_select.csv", index=False)
    assert (out / "model_select.csv").exists()
    assert len(df) == 1


def test_finetune_select_empty_fail_fast(tmp_path):
    repro = tmp_path / "repro"
    manifest_dir = repro / "manifests"
    manifest_dir.mkdir(parents=True)
    pd.DataFrame(columns=["job_id", "status", "result_dir", "source_exp_id"]).to_csv(
        manifest_dir / "pretrain_sweep_manifest.csv", index=False
    )
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{"exp_id": "exp_048", "resolved": True}]).to_csv(resolved, index=False)
    with pytest.raises(RuntimeError, match="No successful Round 9 reproduction"):
        build_round9_finetune_select(str(repro), str(resolved))


def test_analyze_round9_diagnostics_minimal(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame().to_csv(reports / "deconfounding_qc_model_summary.csv", index=False)
    pd.DataFrame().to_csv(reports / "conditional_domain_auc_summary.csv", index=False)
    pd.DataFrame().to_csv(reports / "conditional_domain_auc_by_cancer.csv", index=False)
    pd.DataFrame().to_csv(reports / "prototype_margin_summary.csv", index=False)
    pd.DataFrame().to_csv(reports / "prototype_distance_by_cancer.csv", index=False)
    pd.DataFrame().to_csv(reports / "latent_stability_by_model.csv", index=False)
    agg = tmp_path / "aggregate_scores.csv"
    pd.DataFrame().to_csv(agg, index=False)
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{"exp_id": "exp_048", "resolved": True}]).to_csv(resolved, index=False)
    outdir = tmp_path / "final"
    rc = subprocess.run(
        [sys.executable, "tools/analyze_round9_diagnostics.py",
         "--diagnostics-dir", str(reports),
         "--aggregate", str(agg),
         "--resolved-baselines", str(resolved),
         "--outdir", str(outdir)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    assert (outdir / "round9_final_report.md").exists()
