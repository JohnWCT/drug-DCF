import os
import subprocess


def test_pipeline_script_exists():
    path = "tools/run_round15_repro_rescue_pipeline.sh"
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "round15_config_builder.py" in text
    assert "round15_repro_rescue_qc" in text


def test_config_builder_cli_smoke():
    outdir = "result/optimization_runs/round15_repro_rescue_smoke"
    assert os.path.isfile(os.path.join(outdir, "manifests/finetune_dispatch_manifest.csv"))
