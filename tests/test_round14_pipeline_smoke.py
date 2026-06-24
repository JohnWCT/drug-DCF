#!/usr/bin/env python3
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pipeline_script_exists_and_has_correct_flags():
    path = os.path.join(PROJECT_ROOT, "tools/run_round14_vicreg_stabilizer_pipeline.sh")
    text = open(path, encoding="utf-8").read()
    assert "round14_vicreg_stabilizer_qc" in text
    assert "--mini-batch-size" in text
    assert "--round13-mode" in text
    assert "FINETUNE_PARALLEL" in text
    assert os.access(path, os.X_OK)
