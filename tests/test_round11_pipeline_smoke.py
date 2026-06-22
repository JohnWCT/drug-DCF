"""Smoke tests for Round 11 pipeline wiring."""

import os


def test_run_round11_pipeline_script_exists():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "run_round11_pipeline.sh")
    assert os.path.isfile(path)
    assert os.access(path, os.X_OK)


def test_round11a_qc_module_importable():
    from tools.run_round11a_round10_qc import run_round11a_qc  # noqa: F401
