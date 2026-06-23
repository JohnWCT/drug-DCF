"""Smoke test for Round 12 pipeline wiring."""

import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(
            PROJECT_ROOT,
            "result/optimization_runs/round11_stability_recon/pretrain/exp_035/params.json",
        )
    ),
    reason="Round 11 exp_035 required",
)
def test_config_builder_cli_smoke(tmp_path):
    outdir = tmp_path / "round12_smoke"
    cmd = [
        sys.executable,
        "tools/round12_config_builder.py",
        "--settings",
        "config/round12_proto_alignment_settings.json",
        "--outdir",
        str(outdir),
        "--round11-root",
        "result/optimization_runs/round11_stability_recon",
        "--force",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    manifest = outdir / "manifests" / "pretrain_sweep_manifest.csv"
    assert manifest.is_file()
