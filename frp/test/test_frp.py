##################################################
# Checking if the results of FRP are reproducible
# 1. Fix seed
# 2. Generate several ground truth
# 3. Run FRP for each ground truth
# 4. Run 3 again
# 5. Compare the results of 3 and 4
##################################################

import random
import numpy as np
import sys
import pathlib

CURRENT_DIR = pathlib.Path(__file__).absolute().parent
sys.path = [str(CURRENT_DIR.parent), str(CURRENT_DIR.parent.parent)] + sys.path

from sem_utils import generate_random_data_from_graph
from graph_utils import generate_erdos_renyi_dg, get_edge_num_from_sparsity
from run_causal_discovery import run_filter_rank_prune


def test_run_filter_rank_prune_reproducible():
    # check if running frp is reproducible
    seed = 0
    graph_true_list = [
        generate_erdos_renyi_dg(
            p, get_edge_num_from_sparsity(p, target_sp), seed=seed + seed_inc
        )
        for p in range(10, 11, 2)
        for target_sp in [0.75, 0.5]
        for seed_inc in range(10)
    ]
    n_data = 1000

    # test for 2 random graphs
    random.shuffle(graph_true_list)

    for graph_true in graph_true_list[:2]:
        _, _, X, _ = generate_random_data_from_graph(
            graph_true, n_data, "gaussian", seed
        )
        loss_type = "kld"
        reg_type = "scad"
        reg_params = {"lam": 0.5 * np.log(n_data) / n_data, "gamma": 3.7}
        parcorr_thrs = 0.1
        edge_penalty = 0.5 * np.log(n_data) / n_data
        n_inits = 8  # originally 25, but for test time efficiency
        n_threads = 4  # originally 25, but for test time efficiency
        use_loss_cache = True
        parcorr_filter = True
        loss_inc_tol_multiplier = 4

        res_1 = run_filter_rank_prune(
            X,
            loss_type,
            reg_type,
            reg_params,
            edge_penalty,
            n_inits,
            n_threads,
            parcorr_thrs,
            use_loss_cache,
            seed,
            parcorr_filter,
            loss_inc_tol_multiplier,
            verbose=1,
        )

        res_2 = run_filter_rank_prune(
            X,
            loss_type,
            reg_type,
            reg_params,
            edge_penalty,
            n_inits,
            n_threads,
            parcorr_thrs,
            use_loss_cache,
            seed,
            parcorr_filter,
            loss_inc_tol_multiplier,
            verbose=1,
        )

        assert np.all(res_1["learned_support"] == res_2["learned_support"])


if __name__ == "__main__":
    test_run_filter_rank_prune_reproducible()
