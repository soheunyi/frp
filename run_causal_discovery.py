import numpy.linalg as npla
import numpy as np
import time
import random
import datetime
import multiprocessing as mp


from dglearn.learning.cyclic_manager import CyclicManager
from dglearn.learning.search.tabu import tabu_search
from dglearn.learning.search.virtual import virtual_refine
from dglearn.dg.reduction import reduce_support

from frp.rank_and_prune import RankAndPrune


def run_dglearn(
    X: np.ndarray,
    tabu_length: int,
    patience: int,
    tabu_max_iter: int,
    tabu_move_timeout: int,
    tabu_timeout: int,
    hill_timeout: int,
    refine_timeout: int,
    max_path_len: int,
    force_stable: bool,
    seed: int,
    verbose: int = 0,
    initial_support_seed=None,
    initial_support_density=0.5,
):
    """
    Run DGLEARN on the given data matrix X.
    This is identical to the original DGLEARN code.
    https://github.com/syanga/dglearn
    """
    # learn structure using tabu search, plot learned structure
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()

    manager = CyclicManager(
        X,
        bic_coef=0.5,
        move_timeout=tabu_move_timeout,
        force_stable=force_stable,
        verbose=False,
    )

    initial_support = None
    if initial_support_seed is not None:
        np.random.seed(initial_support_seed)
        random.seed(initial_support_seed)
        initial_support = np.random.choice(
            [0, 1],
            size=(manager.p, manager.p),
            p=[1 - initial_support_density, initial_support_density],
        )
        np.fill_diagonal(initial_support, 0)
        np.random.seed(seed)
        random.seed(seed)

    learned_support, best_score, _ = tabu_search(
        manager,
        tabu_length,
        patience,
        first_ascent=False,
        max_iter=tabu_max_iter,
        tabu_timeout=tabu_timeout,
        hill_timeout=hill_timeout,
        verbose=0,
        initial_support=initial_support,
    )

    # perform virtual edge correction
    if verbose:
        print("virtual edge correction...")
    learned_support = virtual_refine(
        manager,
        learned_support,
        patience=0,
        max_path_len=max_path_len,
        verbose=1,
        timeout=refine_timeout,
    )

    # remove any reducible edges
    if verbose:
        print("reduce support...")
    learned_support = reduce_support(learned_support, fill_diagonal=False)

    dglearn_time = time.time() - start_time

    if verbose:
        print("DGLEARN time: ", dglearn_time)
        print("DGLEARN best score: ", best_score)
        print("DGLEARN edges: ", np.sum(learned_support))
        print("Current time: ", datetime.datetime.now())

    return {
        "learned_support": learned_support,
        "best_score": best_score,
        "time": dglearn_time,
    }


def run_rnp_manager(manager: RankAndPrune):
    manager.rank_and_prune()
    return manager


def run_filter_rank_prune(
    X: np.ndarray,
    loss_type: str,
    reg_type: str,
    reg_params: dict,
    edge_penalty: float,
    n_inits: int,
    n_threads: int,
    parcorr_thrs: float,
    use_loss_cache: bool,
    seed: int,
    parcorr_filter: bool = True,
    loss_inc_tol_multiplier: float = 4,
    verbose: int = 0,
    use_rank: bool = True,
    use_one_edge_removal: bool = True,
):
    np.random.seed(seed)
    random.seed(seed)
    _, p = X.shape

    start_time = time.time()

    #################################################################
    ######################## Filter Stage ###########################
    #################################################################

    COV_MLE = X.T @ X / X.shape[0]
    THETA_MLE = npla.inv(COV_MLE)

    parcorr_est = -THETA_MLE / (
        np.outer(THETA_MLE.diagonal(), THETA_MLE.diagonal()) ** 0.5
    )
    if parcorr_filter:
        supp_THETA = np.abs(parcorr_est) > parcorr_thrs
    else:
        supp_THETA = np.ones_like(parcorr_est, dtype=bool)
    THETA_input = THETA_MLE

    #################################################################
    ######################## Rank and Prune #########################
    #################################################################

    rnp_managers = [
        RankAndPrune(
            THETA_input,
            supp_THETA,
            loss_type,
            reg_type,
            reg_params,
            edge_penalty,
            loss_inc_tol=loss_inc_tol_multiplier * edge_penalty,
            search="binary",
            verbose=0,
            use_loss_cache=use_loss_cache,
            seed=seed,
            use_rank=use_rank,
            use_one_edge_removal=use_one_edge_removal,
        )
        for seed in range(n_inits)
    ]

    with mp.Pool(n_threads) as pool:
        rnp_managers = pool.map(run_rnp_manager, rnp_managers)

    assert all([r.rnp_exec_flag for r in rnp_managers])

    Q_stack = np.array([r.supp for r in rnp_managers])
    scores = np.array([r.score for r in rnp_managers])

    end_time = time.time()

    ####################################################################
    ################## Evaluate and Save results #######################
    ####################################################################

    supp_W_ests = (Q_stack != 0).astype(int) * (np.eye(p) == 0).astype(int)
    supp_W_ests = supp_W_ests.astype(bool)

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_supp_W_est = supp_W_ests[best_idx]
    best_edges = best_supp_W_est.sum()

    frp_time = end_time - start_time

    if verbose:
        print("FRP time: ", frp_time)
        print("FRP edges: ", best_edges)
        print("FRP score: ", best_score)
        print("Current time: ", datetime.datetime.now())

    return {
        "learned_support": best_supp_W_est,
        "best_score": best_score,
        "time": frp_time,
    }
