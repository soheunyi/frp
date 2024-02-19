import random
from typing import Literal
import numpy as np
from dglearn.evaluation.evaluation_kld import minimize_kld
from estimate_loss import estimate_loss
import matplotlib.pyplot as plt


def refine_edges(
    THETA: np.ndarray,
    A: np.ndarray,
    lam: float,
    verbose=False,
    method: Literal["sequential", "binary"] = "binary",
    objective: Literal["kld", "kld_2nd_approx", "l2"] = "kld_2nd_approx",
):
    if objective not in ["kld", "kld_2nd_approx", "l2"]:
        raise ValueError("objective must be one of kld, kld_2nd_approx, l2")

    def _minimizer(THETA, A):
        return estimate_loss(objective, "zero", {}, THETA, A, "L-BFGS-B")

    def _edges_sorted_Q(Q, A):
        edges_list = np.argwhere(A).tolist()
        # sort ascending by Q value
        edges_list.sort(key=lambda x: abs(Q[x[0], x[1]]))
        return edges_list

    A_cpy = A.copy()
    kld, Q = _minimizer(THETA, A_cpy)
    loss = kld + lam * np.sum(A_cpy)
    while True:
        removed = False
        edges_list = _edges_sorted_Q(Q, A_cpy)

        if verbose:
            print([Q[i, j] for i, j in edges_list])

        for idx, (i, j) in enumerate(edges_list):
            A_tmp = A_cpy.copy()
            A_tmp[i, j] = False
            new_kld, new_Q = _minimizer(THETA, A_tmp, Q_init=Q, thresh=kld + 0.5 * lam)
            loss_new = new_kld + lam * np.sum(A_tmp)
            if loss_new < loss:
                if verbose:
                    print("Removed edge ({}, {})".format(i, j))
                    print("Removed order = {}".format(idx))
                    print("Removed Q value = {}".format(Q[i, j]))
                A_cpy = A_tmp.copy()
                loss = loss_new
                kld = new_kld
                Q = new_Q
                removed = True
                break

        if not removed:
            break

    if verbose:
        print("Final # edges = {}".format(np.sum(A_cpy)))
        print("Final loss = {}".format(loss))
        print("Final kld = {}".format(kld))

    return A_cpy
