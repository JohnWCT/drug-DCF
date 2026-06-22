"""Smoke tests for Round 11 pipeline wiring."""

import os
import re


def test_run_round11_pipeline_script_exists():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "run_round11_pipeline.sh")
    assert os.path.isfile(path)
    assert os.access(path, os.X_OK)


def test_round11a_qc_module_importable():
    from tools.run_round11a_round10_qc import run_round11a_qc  # noqa: F401


def test_pretrain_passes_reconstruction_loss_kwargs_to_train_d_ae():
    root = os.path.dirname(os.path.dirname(__file__))
    src = open(os.path.join(root, "pretrain_VAEwC.py"), encoding="utf-8").read()
    assert "recon_loss_kw = reconstruction_loss_kwargs(param)" in src
    assert "reconstruction_loss_kwargs=recon_loss_kw" in src
    assert re.search(r"vaeloss\([^)]+\*\*recon_loss_kw", src)


def test_run_summary_includes_reconstruction_loss_block():
    root = os.path.dirname(os.path.dirname(__file__))
    src = open(os.path.join(root, "pretrain_VAEwC.py"), encoding="utf-8").read()
    assert '"reconstruction_loss": reconstruction_loss_payload' in src
