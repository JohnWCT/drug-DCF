import json
from pathlib import Path

from tools.round18_feature_coverage import assert_round18_feature_coverage


def test_feature_coverage_identical_modelids():
    settings = json.loads(Path('config/round18_architecture_settings.json').read_text())
    report = assert_round18_feature_coverage(settings)
    assert report['ok'] is True
    assert report['model_id_sets_identical'] is True
    assert report['n_model_ids'] > 0
