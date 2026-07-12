import json
from pathlib import Path

from step1_finetune_latent_pipeline_round18_cv import run_smoke


class _Args:
    outdir = "result/optimization_runs/round18_architecture"
    steps = 1


def test_pipeline_smoke_entrypoint(tmp_path):
    args = _Args()
    args.outdir = str(tmp_path / "r18")
    summary = run_smoke(args)
    assert summary["ok"] is True
    assert summary["n_architectures"] == 4
    assert Path(summary["resource_dir"], "runtime_resource_summary.json").exists()
