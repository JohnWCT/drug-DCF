from biocda.models.biocda_model import BioCDA
from biocda.models.model_factory import build_model
from biocda.models.pooled_baseline import PooledBaselineModel
import pytest

def test_build_cross_attention_model(xa_config):
    assert isinstance(build_model(xa_config), BioCDA)

def test_build_pooled_baseline(pooled_config):
    assert isinstance(build_model(pooled_config), PooledBaselineModel)

def test_unknown_model_type_fails(xa_config):
    bad = dict(xa_config)
    bad["model"] = dict(xa_config["model"])
    bad["model"]["type"] = "unknown"
    with pytest.raises(ValueError):
        build_model(bad)
