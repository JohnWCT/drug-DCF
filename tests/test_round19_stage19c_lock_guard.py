"""Formal selection lock must remain NO-GO until 19C completes."""

from __future__ import annotations

import subprocess
import sys


def test_formal_selection_lock_refused():
    proc = subprocess.run(
        [
            sys.executable,
            "tools/analyze_round19.py",
            "--stage",
            "selection",
            "--write-lock",
            "--outdir",
            "result/optimization_runs/round19_factorial",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert proc.returncode != 0
    assert "Refuse lock" in (proc.stderr + proc.stdout)


def test_formal_selection_requires_write_lock_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "tools/analyze_round19.py",
            "--stage",
            "selection",
            "--outdir",
            "result/optimization_runs/round19_factorial",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert proc.returncode != 0
    assert "requires --write-lock" in (proc.stderr + proc.stdout)
