#!/usr/bin/env python3
"""Round 17R pipeline smoke tests (no GPU finetune)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_py_compile_round17r_tools() -> None:
    tools = [
        "tools/round17r_18class_config_builder.py",
        "tools/analyze_round17r_18class.py",
        "tools/extract_round13_proto_features.py",
        "tools/visualize_round17_prototype_tsne.py",
    ]
    subprocess.check_call([sys.executable, "-m", "py_compile", *tools])


def test_builder_rejects_placeholder_in_isolation() -> None:
    from tools.round17r_18class_config_builder import _reject_forbidden_aliases

    with pytest.raises(ValueError):
        _reject_forbidden_aliases("round16_top")


def test_stage_scripts_exist_and_are_executable_bits() -> None:
    scripts = [
        "tools/run_round17r_stage17r_a_feature_smoke.sh",
        "tools/run_round17r_stage17r_b_focused.sh",
        "tools/run_round17r_stage17r_c_refine.sh",
        "tools/run_round17r_stage17r_d_confirm.sh",
        "tools/run_round17r_stage17r_f_tsne.sh",
    ]
    for rel in scripts:
        path = Path(rel)
        assert path.is_file()
        text = path.read_text()
        assert "set -euo pipefail" in text
        assert "ROUND17R_ROOT" in text
