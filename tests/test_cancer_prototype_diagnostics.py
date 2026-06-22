import numpy as np
from tools.round9_diagnostics_common import cosine_distance, euclidean_distance

def test_cosine_distance_zero_for_same_vector():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == 0.0

def test_euclidean_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == 5.0
