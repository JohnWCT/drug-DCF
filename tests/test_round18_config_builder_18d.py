import pytest

from tools.round18_config_builder import build_stage18d_manifest, load_json


def test_18d_fails_without_lock(tmp_path):
    settings = load_json('config/round18_architecture_settings.json')
    with pytest.raises(FileNotFoundError):
        build_stage18d_manifest(settings, str(tmp_path / 'r18'), allow_placeholder=False)


def test_18d_placeholder_opt_in(tmp_path):
    settings = load_json('config/round18_architecture_settings.json')
    out = build_stage18d_manifest(settings, str(tmp_path / 'r18'), allow_placeholder=True)
    assert out['lock_source'] == 'placeholder_until_analyzer'
    assert out['n_jobs'] == 20
