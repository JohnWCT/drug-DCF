import json
from tests.round9_test_helpers import write_minimal_checkpoint
from tools.round9_baseline_resolver import resolve_baselines

def test_resolve_explicit_path(tmp_path):
    exp_dir = write_minimal_checkpoint(str(tmp_path), "exp_048")
    cfg = tmp_path / "baselines.json"
    payload = {"baselines": [{"exp_id": "exp_048", "role": "primary", "required": True, "explicit_path": str(exp_dir)}]}
    cfg.write_text(json.dumps(payload))
    resolved, missing, _, code = resolve_baselines(str(cfg), search_root=str(tmp_path))
    assert code == 0
    assert len(resolved) == 1

def test_required_missing_fail_fast(tmp_path):
    cfg = tmp_path / "baselines.json"
    payload = {"baselines": [{"exp_id": "exp_not_in_hints", "role": "primary", "required": True, "explicit_path": None}]}
    cfg.write_text(json.dumps(payload))
    _, missing, _, code = resolve_baselines(str(cfg), search_root=str(tmp_path))
    assert code == 2
    assert "exp_not_in_hints" in set(missing["exp_id"])

def test_optional_missing_does_not_fail(tmp_path):
    exp_dir = write_minimal_checkpoint(str(tmp_path), "exp_048")
    cfg = tmp_path / "baselines.json"
    payload = {"baselines": [
        {"exp_id": "exp_048", "role": "primary", "required": True, "explicit_path": str(exp_dir)},
        {"exp_id": "exp_missing", "role": "optional", "required": False, "explicit_path": None},
    ]}
    cfg.write_text(json.dumps(payload))
    resolved, missing, _, code = resolve_baselines(str(cfg), search_root=str(tmp_path))
    assert code == 0
    assert len(resolved) == 1
    assert "exp_missing" in set(missing["exp_id"])
