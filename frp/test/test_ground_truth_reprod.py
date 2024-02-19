##################################################
# Testing we are generating identical ground truth
# data for the same graph and seed
##################################################

import pathlib
import sys

sys.path = [str(pathlib.Path(__file__).absolute().parent.parent)] + sys.path
from sem_utils import (
    generate_random_data_from_graph,
    generate_stable_random_data_from_graph,
)
from graph_utils import generate_erdos_renyi_dg, get_edge_num_from_sparsity


def test_generate_ground_truth():
    # check if the ground truth data generation is reproducible
    seed = 0
    graph_true_list = [
        generate_erdos_renyi_dg(
            p, get_edge_num_from_sparsity(p, target_sp), seed=seed + seed_inc
        )
        for p in range(10, 21, 2)
        for target_sp in [0.75, 0.5, 0.25, 0.125]
        for seed_inc in range(10)
    ]
    n_data_list = [500, 1000, 5000, 10000]

    for graph_true in graph_true_list:
        W_true_list = []
        sigmas_list = []
        THETA_list = []
        for n_data in n_data_list:
            W_true, sigmas, X, THETA = generate_random_data_from_graph(
                graph_true, n_data, "gaussian", seed
            )
            W_true_list.append(W_true)
            sigmas_list.append(sigmas)
            THETA_list.append(THETA)

        assert (W_true_list[0] == W_true_list[1]).all()
        assert (W_true_list[0] == W_true_list[2]).all()
        assert (W_true_list[0] == W_true_list[3]).all()
        assert (sigmas_list[0] == sigmas_list[1]).all()
        assert (sigmas_list[0] == sigmas_list[2]).all()
        assert (sigmas_list[0] == sigmas_list[3]).all()
        assert (THETA_list[0] == THETA_list[1]).all()
        assert (THETA_list[0] == THETA_list[2]).all()
        assert (THETA_list[0] == THETA_list[3]).all()

    for graph_true in graph_true_list:
        W_true_list = []
        sigmas_list = []
        THETA_list = []
        for n_data in n_data_list:
            W_true, sigmas, X, THETA = generate_stable_random_data_from_graph(
                graph_true, n_data, "gaussian", seed
            )
            W_true_list.append(W_true)
            sigmas_list.append(sigmas)
            THETA_list.append(THETA)

        assert (W_true_list[0] == W_true_list[1]).all()
        assert (W_true_list[0] == W_true_list[2]).all()
        assert (W_true_list[0] == W_true_list[3]).all()
        assert (sigmas_list[0] == sigmas_list[1]).all()
        assert (sigmas_list[0] == sigmas_list[2]).all()
        assert (sigmas_list[0] == sigmas_list[3]).all()
        assert (THETA_list[0] == THETA_list[1]).all()
        assert (THETA_list[0] == THETA_list[2]).all()
        assert (THETA_list[0] == THETA_list[3]).all()

    for graph_true in graph_true_list:
        X_list = []
        for _ in range(2):
            X = generate_random_data_from_graph(graph_true, 1000, "gaussian", seed)[2]
            X_list.append(X)

        assert (X_list[0] == X_list[1]).all()
