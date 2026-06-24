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


def test_optimization_runner_accepts_round15_selection_mode():
    from tools.optimization_runner import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "select",
        "--run-dir", "result/optimization_runs/round15_repro_rescue_smoke",
        "--result-dir", "result/optimization_runs/round15_repro_rescue_smoke/pretrain",
        "--selection-mode", "round15_repro_rescue_qc",
    ])

    assert args.selection_mode == "round15_repro_rescue_qc"
