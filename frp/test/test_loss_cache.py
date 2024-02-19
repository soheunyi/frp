##################################################
# Testing LossCache implementation
##################################################

import numpy as np
import pathlib
import sys

sys.path = [str(pathlib.Path(__file__).absolute().parent.parent)] + sys.path
from loss_cache import LossCache, including_indices


def test_including_indices():
    cached_adjs = np.array(
        [
            [[False, True, False], [True, False, True], [False, False, False]],
            [[False, True, False], [True, False, False], [False, False, True]],
            [[False, True, False], [False, False, True], [False, False, True]],
        ]
    )

    adj = np.array([[False, True, False], [False, False, False], [False, False, False]])
    assert np.all(including_indices(cached_adjs, adj) == np.array([0, 1, 2]))

    adj = np.array([[False, True, True], [False, False, False], [False, False, False]])
    assert np.all(including_indices(cached_adjs, adj) == np.array([]))

    adj = cached_adjs[0] & cached_adjs[1]
    assert np.all(including_indices(cached_adjs, adj) == np.array([0, 1]))

    adj = cached_adjs[0] & cached_adjs[2]
    assert np.all(including_indices(cached_adjs, adj) == np.array([0, 2]))

    adj = cached_adjs[1] & cached_adjs[2]
    assert np.all(including_indices(cached_adjs, adj) == np.array([1, 2]))

    adj = cached_adjs[1] & cached_adjs[2] & cached_adjs[0]
    assert np.all(including_indices(cached_adjs, adj) == np.array([0, 1, 2]))


def test_loss_cache():
    A1 = np.array([[False, True, False], [True, False, True], [False, False, False]])
    A2 = np.array([[False, True, False], [True, False, False], [True, False, False]])
    A3 = np.array([[False, True, False], [False, False, True], [False, False, False]])

    loss_cache = LossCache(3)

    loss_cache.add(A1, 1)
    assert np.all(loss_cache.adjs == A1)

    loss_cache.add(A2, 2)
    assert np.all(loss_cache.adjs == np.concatenate([[A1], [A2]], axis=0))

    assert loss_cache.get_loss_lb(A1 & A2) == 2

    loss_cache.add(A3, 3)
    assert np.all(loss_cache.adjs == np.concatenate([[A1], [A2], [A3]], axis=0))
    assert loss_cache.get_loss_lb(A1 & A2 & A3) == 3

    loss_cache.add(A1, 4)
    assert np.all(loss_cache.adjs == np.concatenate([[A1], [A2], [A3]], axis=0))
    assert loss_cache.get_loss_lb(A1 & A2 & A3) == 4
    assert loss_cache.get_loss_lb(A1 & A2) == 4
