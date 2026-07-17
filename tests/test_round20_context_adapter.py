from __future__ import annotations

import json
from pathlib import Path

from tools.round20_context_adapter import audit_context_pair, inspect_context_dir

ROOT = Path(__file__).resolve().parents[1]
C16 = ROOT / "result/optimization_runs/round19_factorial/features/z_plus_context16"


def test_inspect_c16_ok() -> None:
    report = inspect_context_dir(C16, expected_context_dim=16)
    assert report["ok"] is True
    assert report["component_slices"]["output_dim"] == 80
    assert report["component_slices"]["latent_slice"] == [0, 64]
    assert report["component_slices"]["context_slice"] == [64, 80]


def test_audit_reports_missing_c32(tmp_path: Path) -> None:
    out = tmp_path / "context_audit.json"
    report = audit_context_pair(c16_dir=C16, c32_dir=None, out=out, fail_closed=False)
    assert report["c16"]["ok"] is True
    assert report["c32"]["ok"] is False
    assert report["comparable"] is False
    assert "c32_not_ok" in report["mismatches"]
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rebuild_guidance"]["auto_rebuild_in_audit"] is False
