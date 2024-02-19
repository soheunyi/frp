import numpy as np


def including_indices(cached_adjs: np.ndarray, adj: np.ndarray) -> list:
    # return indices of cached_adjs that include adj
    if cached_adjs.size == 0:
        return False
    assert cached_adjs.shape[1:] == adj.shape
    return np.where((adj & ~cached_adjs).sum(axis=(1, 2)) == 0)[0]


class LossCache:
    def __init__(self, p: int):
        self.p = p
        self.adjs = np.array([], dtype=bool)
        self.losses = np.array([], dtype=float)

    def check_adj(self, adj: np.ndarray) -> bool:
        assert adj.shape[0] == adj.shape[1] == self.p
        assert np.all(adj.diagonal() == 0)

    def add(self, adj: np.ndarray, loss: float):
        self.check_adj(adj)
        # if self.adjs is empty, initialize it
        if self.adjs.size == 0:
            self.adjs = adj[np.newaxis, :, :]
            self.losses = np.array([loss])
        else:
            # if adj is new, stack it into self.adjs
            adj_copys = np.repeat(adj[np.newaxis, :, :], self.adjs.shape[0], axis=0)
            identical = np.all(self.adjs == adj_copys, axis=(1, 2))
            if not np.any(identical):
                self.adjs = np.concatenate([self.adjs, adj[np.newaxis, :, :]], axis=0)
                self.losses = np.concatenate([self.losses, [loss]])
            else:
                # if adj is not new, update the loss
                self.losses[identical] = loss

    def get_loss_lb(self, adj: np.ndarray) -> float:
        self.check_adj(adj)
        # get loss lower bound for adj
        if self.adjs.size == 0:
            return 0

        incl_indices = including_indices(self.adjs, adj)
        if incl_indices.size == 0:
            return 0
        else:
            return self.losses[incl_indices].max()


# test code
if __name__ == "__main__":

    def test_including_indices():
        cached_adjs = np.array(
            [
                [[False, True, False], [True, False, True], [False, False, False]],
                [[False, True, False], [True, False, False], [False, False, True]],
                [[False, True, False], [False, False, True], [False, False, True]],
            ]
        )

        adj = np.array(
            [[False, True, False], [False, False, False], [False, False, False]]
        )
        assert np.all(including_indices(cached_adjs, adj) == np.array([0, 1, 2]))

        adj = np.array(
            [[False, True, True], [False, False, False], [False, False, False]]
        )
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
        A1 = np.array(
            [[False, True, False], [True, False, True], [False, False, False]]
        )
        A2 = np.array(
            [[False, True, False], [True, False, False], [True, False, False]]
        )
        A3 = np.array(
            [[False, True, False], [False, False, True], [False, False, False]]
        )

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

    test_including_indices()
    test_loss_cache()
