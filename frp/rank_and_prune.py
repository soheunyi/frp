import random
import time
from typing import Literal, Optional
import numpy as np

from estimate_loss import LossEstimator
from loss_cache import LossCache


def sort_edges_by_absQ(Q, A):
    edges_list = np.argwhere(A).tolist()
    # sort ascending by Q value
    edges_list.sort(key=lambda x: abs(Q[x[0], x[1]]))
    return edges_list


def random_edges(A):
    edges_list = np.argwhere(A).tolist()
    random.shuffle(edges_list)
    return edges_list


def get_adj_from_supp(supp):
    return (supp != 0) & (np.eye(supp.shape[0]) == 0)


def get_augmented_support(edges, p):
    supp = np.eye(p)
    for i, j in edges:
        supp[i, j] = 1
    return supp


def get_n_edges(supp):
    return np.sum(get_adj_from_supp(supp))


def remove_support(supp, rmv_edges):
    supp_del = supp.copy()
    for i, j in rmv_edges:
        supp_del[i, j] = 0
    return supp_del


class RankAndPrune:
    def __init__(
        self,
        THETA,
        init_support,
        loss_type,
        reg_type,
        reg_params,
        edge_penalty,
        loss_inc_tol,
        search: Literal["sequential", "binary"] = "sequential",
        verbose=0,
        use_loss_cache: bool = True,
        seed: Optional[int] = None,
        use_rank: bool = True,
        use_one_edge_removal: bool = True,
    ):
        """
        verbose = 0: no output
        verbose = 1: only summary
        verbose = 2: print # edges and loss after each deletion step
        """
        assert verbose in [0, 1, 2]
        assert search in ["sequential", "binary"]

        self.THETA = THETA
        self.loss_type = loss_type
        self.reg_type = reg_type
        self.reg_params = reg_params
        self.edge_penalty = edge_penalty
        self.loss_inc_tol = loss_inc_tol
        self.search = search
        self.verbose = verbose
        self.loss_cache = LossCache(self.THETA.shape[0]) if use_loss_cache else None
        self.init_support = init_support
        self.supp = init_support
        self.seed = seed
        self.use_rank = use_rank
        self.use_one_edge_removal = use_one_edge_removal

        self.minimizer_call_count = 0
        self.approx_flops = 0

        self.rnp_exec_flag = False
        self.score = None
        self.rnp_loss_inc_history = []

        self.no_penalty_loss_estimator = LossEstimator(
            self.THETA, self.loss_type, "zero", {}
        )
        self.reg_loss_estimator = LossEstimator(
            self.THETA, self.loss_type, self.reg_type, self.reg_params
        )

    def update_cache(self, adj, loss):
        if self.loss_cache is not None:
            self.loss_cache.add(adj, loss)

    def summarize(self, supp):
        vl, _ = self.no_penalty_loss_estimator.estimate_loss(supp, -np.inf)
        print("loss:", vl)
        print("# edges:", get_n_edges(supp))

    def rank_and_prune_once(self):
        def probe(edges: list, loss_ub, loss_theo_best, verbose):
            if verbose >= 2:
                print("# del edges:", len(edges))
            supp = remove_support(self.supp, edges)
            supp_adj = get_adj_from_supp(supp)
            if (
                self.loss_cache is not None
                and self.loss_cache.get_loss_lb(supp_adj) > loss_ub
            ):
                if verbose >= 2:
                    print("loss_lb > loss_ub")
                return True

            v_loss, _ = self.no_penalty_loss_estimator.estimate_loss(
                supp, loss_theo_best
            )
            self.update_cache(supp_adj, v_loss)

            if verbose >= 2:
                print("loss:", v_loss)

            return v_loss > loss_ub

        loss_theo_best, Q = self.no_penalty_loss_estimator.estimate_loss(
            self.supp, -np.inf
        )
        self.update_cache(get_adj_from_supp(self.supp), loss_theo_best)
        loss_ub = loss_theo_best + self.loss_inc_tol

        if self.verbose >= 1:
            print("loss_theo_best:", loss_theo_best)

        # Perform penalized minimization

        if self.use_rank:
            _, Q = self.reg_loss_estimator.estimate_loss(self.supp, loss_theo_best)
            edges_list = sort_edges_by_absQ(Q, get_adj_from_supp(self.supp))
        else:
            edges_list = random_edges(get_adj_from_supp(self.supp))

        # search space: n_del \in range(1, len(edges_list))
        if self.search == "binary":
            n_del_lb = 1
            n_del_ub = len(edges_list) - 1
            n_del = len(edges_list) // 2
            # binary search
            while n_del_lb < n_del_ub:
                if probe(edges_list[:n_del], loss_ub, loss_theo_best, self.verbose):
                    n_del_ub = n_del
                else:
                    n_del_lb = n_del

                if n_del_ub == n_del_lb + 1:
                    break
                else:
                    n_del = (n_del_lb + n_del_ub) // 2

            n_del = n_del_ub

            if n_del_lb == 1:
                if probe(edges_list[:1], loss_ub, loss_theo_best, self.verbose):
                    n_del = 1
        elif self.search == "sequential":
            # sequential search
            for n_del in range(1, len(edges_list)):
                if probe(edges_list[:n_del], loss_ub, loss_theo_best, self.verbose):
                    break
        else:
            raise ValueError(f"search method {self.search} not supported")

        if n_del > 1:
            pruned_supp = remove_support(self.supp, edges_list[: n_del - 1])
        else:
            if not self.use_one_edge_removal:
                pruned_supp = self.supp.copy()
                return pruned_supp

            # try deleting other edge if n_del == 1,
            # i.e., deleting the first edge results in surpassing loss_ub
            if self.verbose >= 1:
                print("trying deleting other edge...")
            deleted = False
            for e_idx in range(1, len(edges_list)):
                if self.verbose >= 2:
                    print("trying deleting edge", edges_list[e_idx])

                if not probe(
                    edges_list[e_idx : e_idx + 1], loss_ub, loss_theo_best, self.verbose
                ):
                    deleted = True
                    break

            if deleted:
                pruned_supp = remove_support(self.supp, edges_list[e_idx : e_idx + 1])
            else:
                pruned_supp = self.supp.copy()

        if self.verbose >= 1:
            self.summarize(pruned_supp)

        n_removed_edges = np.sum(get_adj_from_supp(self.supp)) - np.sum(
            get_adj_from_supp(pruned_supp)
        )
        loss_inc = (
            self.no_penalty_loss_estimator.estimate_loss(pruned_supp, loss_theo_best)[0]
            - loss_theo_best
        )
        self.rnp_loss_inc_history.append((n_removed_edges, loss_inc, pruned_supp))

        return pruned_supp

    def rank_and_prune(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # pruning
        if self.verbose >= 1:
            print("Starting pruning...")
        while True:
            st = time.time()
            last_supp = self.supp.copy()
            self.supp = self.rank_and_prune_once()
            en = time.time()
            if self.verbose >= 1:
                print("Time for pruning:", en - st)
                if self.loss_cache is not None:
                    print("loss cache size:", self.loss_cache.losses.shape)

            # less time cost operation
            if np.all(self.supp == last_supp):
                break

        # update score
        self.eval_score()
        self.rnp_exec_flag = True
        self.minimizer_call_count = (
            self.no_penalty_loss_estimator.call_count
            + self.reg_loss_estimator.call_count
        )
        self.approx_flops = (
            self.no_penalty_loss_estimator.approx_flops
            + self.reg_loss_estimator.approx_flops
        )

    def eval_score(self):
        score, _ = self.no_penalty_loss_estimator.estimate_loss(self.supp)
        n_edges = get_n_edges(self.supp)
        score += self.edge_penalty * n_edges
        self.score = score
