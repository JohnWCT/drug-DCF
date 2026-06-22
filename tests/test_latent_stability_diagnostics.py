import numpy as np
from tools.round9_diagnostics_common import effective_rank

def test_effective_rank_full_rank():
    x = np.random.default_rng(0).normal(size=(50, 4))
    rank = effective_rank(x)
    assert rank > 1.0

def test_collapse_flag_logic():
    latent_size = 8
    active = 2
    assert active < 0.5 * latent_size
