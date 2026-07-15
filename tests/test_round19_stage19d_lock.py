"""Guards for Round 19D / formal selection locks."""
from __future__ import annotations
import pytest
from tools.analyze_round19 import main as analyze_main
import sys

def test_formal_selection_still_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_round19.py", "--stage", "selection", "--write-lock"])
    with pytest.raises(SystemExit):
        analyze_main()
