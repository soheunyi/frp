import random
import igraph as ig
import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from typing import Literal, Optional
from graph_utils import (
    generate_erdos_renyi_dg,
    generate_sparse_graph,
    generate_uniform_graph,
)
from utils import check_numpy

# random SEM generation


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a graph.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of graph
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of graph
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.rand(*B.shape) * (high - low) + low
        W += B * (S == i) * U
    return W


def generate_weight(graph: ig.Graph, seed: int, w_ranges=((-1.0, -0.6), (0.6, 1.0))):
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    B_true = np.array(graph.get_adjacency().data)
    W_true = simulate_parameter(B_true, w_ranges=w_ranges)

    return W_true


def generate_random_data(
    W_true: np.ndarray,
    p: int,
    n: int,
    sigmas: np.ndarray,
    noise_type: Literal["gaussian", "uniform", "exp", "laplace"],
    seed: Optional[int] = None,
):
    """
    random data generation
    input : W_true, p, n, sigmas, noise_type
    output : X
    """

    check_numpy([W_true, sigmas])

    assert len(sigmas) == p
    SIGMA = np.diag(sigmas)

    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    if noise_type == "gaussian":
        std_noise = np.random.randn(n, p)
    elif noise_type == "uniform":
        std_noise = (12**0.5) * (np.random.rand(n, p) - (1 / 2))
    elif noise_type == "exp":
        std_noise = np.random.exponential(1, size=(n, p)) - 1
    elif noise_type == "laplace":
        std_noise = np.random.laplace(0, 1, size=(n, p)) / np.sqrt(2)
    else:
        raise ValueError(
            f"noise_type should be in ['gaussian', 'uniform', 'exp', 'laplace'], got '{noise_type}'."
        )

    X = std_noise @ (SIGMA**0.5) @ npla.inv(np.eye(p) - W_true)

    return X


def generate_stable_random_data_from_graph(
    graph_true: ig.Graph, n_data: int, noise_type: str, seed: int
):
    seed_inc = 0

    np.random.seed(seed)
    random.seed(seed)

    while True:
        W_true = generate_weight(
            graph_true, seed=seed + seed_inc, w_ranges=((-0.8, -0.2), (0.2, 0.8))
        )
        p = W_true.shape[0]
        sigmas = 1 + 2 * np.random.rand(p)

        THETA = (np.eye(p) - W_true) @ np.diag(1 / sigmas) @ (np.eye(p) - W_true).T
        THETA_eigs = npla.eig(THETA)[0]
        W_true_eigs = npla.eig(W_true)[0]

        # calculate max modulus of W_true_eigs
        W_true_eigs_mod = np.abs(W_true_eigs)
        W_true_eigs_mod_max = W_true_eigs_mod.max()

        if (
            W_true_eigs_mod_max < 1
            and 1e-3 < THETA_eigs.real.min() < THETA_eigs.real.max() < 1e3
        ):
            break
        else:
            if seed != None:
                seed_inc += 1000
                np.random.seed(seed + seed_inc)
                random.seed(seed + seed_inc)

    X = generate_random_data(
        W_true, p, n_data, sigmas, noise_type, seed=seed + seed_inc
    )

    np.random.seed(seed)
    random.seed(seed)

    return W_true, sigmas, X, THETA


def generate_random_data_from_graph(
    graph_true: ig.Graph, n_data: int, noise_type: str, seed: int
):
    seed_inc = 0

    np.random.seed(seed)
    random.seed(seed)

    while True:
        W_true = generate_weight(graph_true, seed=seed + seed_inc)
        p = W_true.shape[0]
        sigmas = 1 + np.random.rand(p)

        THETA = (np.eye(p) - W_true) @ np.diag(1 / sigmas) @ (np.eye(p) - W_true).T
        THETA_eigs = npla.eig(THETA)[0]

        if 1e-3 < THETA_eigs.real.min() < THETA_eigs.real.max() < 1e3:
            break
        else:
            if seed != None:
                seed_inc += 1000
                np.random.seed(seed + seed_inc)
                random.seed(seed + seed_inc)

    X = generate_random_data(
        W_true, p, n_data, sigmas, noise_type, seed=seed + seed_inc
    )

    np.random.seed(seed)
    random.seed(seed)

    return W_true, sigmas, X, THETA


def generate_suitable_ground_truth(
    seed: Optional[int], sem_args: dict, verbose: bool = False, n_data: int = 1000
):
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    seed_inc = 0
    p = sem_args["graph"]["p"]

    while True:
        gen_mode = sem_args["graph"]["gen_mode"]
        if gen_mode in ["in", "out"]:
            graph_true = generate_uniform_graph(p, sem_args["graph"]["deg"], gen_mode)
        elif gen_mode == "sparse":
            graph_true = generate_sparse_graph(p, sem_args["graph"]["edge_prob"])
        elif gen_mode == "erdos_renyi":
            graph_true = generate_erdos_renyi_dg(p, sem_args["graph"]["edge_num"])
        else:
            raise ValueError(
                f"gen_mode should be in ['in', 'out', 'sparse', 'erdos_renyi']."
            )
        W_true = generate_weight(graph_true, seed=seed + seed_inc)
        sigmas = 1 + np.random.rand(p)

        X = generate_random_data(W_true, p, n_data, sigmas, sem_args["noise_type"])
        THETA = (np.eye(p) - W_true) @ np.diag(1 / sigmas) @ (np.eye(p) - W_true).T

        dev = X - np.mean(X, axis=0)
        THETA_MLE = npla.inv((dev.T @ dev) / n_data)
        THETA_MLE_eigs = npla.eig(THETA_MLE)[0]

        if THETA_MLE_eigs.real.min() > 1e-3:
            break
        else:
            if seed != None:
                seed_inc += 1000
                np.random.seed(seed + seed_inc)
            if verbose:
                print("Seed changed because THETA_MLE is not Cholesky decomposable.")

    np.random.seed(seed)

    return graph_true, W_true, sigmas, X, THETA


def check_dag(W: np.array):
    return np.trace(spla.expm(np.abs(W))) - W.shape[0] == 0
