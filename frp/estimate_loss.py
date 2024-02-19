import numpy as np
import scipy.optimize as sopt
from converters import y_to_Q, grad_Q_to_grad_y
from loss_utils import LossManager
from reg_utils import RegManager


class LossEstimator:
    def __init__(
        self,
        THETA,
        loss_type,
        reg_type,
        reg_params,
        tol=1e-6,
        max_iters=50,
        patience=10,
        init_generator=None,
        verbose=False,
    ):
        self.loss_manager = LossManager(THETA)
        self.loss_type = loss_type

        self.reg_manager = RegManager(**reg_params)
        self.reg_type = reg_type

        self.tol = tol
        self.max_iters = max_iters
        self.patience = patience
        self.init_generator = init_generator
        self.verbose = verbose

        self.call_count = 0
        self.approx_flops = 0

    def loss_fn(self, Q: np.ndarray):
        _loss_fn = self.loss_manager.__getattribute__(self.loss_type)()
        loss, g_loss = _loss_fn(Q @ Q.T)
        g_Q = 2 * g_loss @ Q

        return loss, g_Q

    def reg_fn(self, Q: np.ndarray):
        _reg_fn = self.reg_manager.__getattribute__(self.reg_type)()
        p = Q.shape[0]
        v, g_Q = _reg_fn(Q * (1 - np.eye(p)))

        return v, g_Q * (1 - np.eye(p))

    def obj_fn(self, Q: np.ndarray):
        vl, g_Ql = self.loss_fn(Q)
        vr, g_Qr = self.reg_fn(Q)
        return vl + vr, g_Ql + g_Qr

    def estimate_loss(self, B_support, theo_best=-np.inf):
        """
        Find parameters for a fixed graph structure (B_support) in order to minimize KLD
        with the ground truth distribution (precision matrix prec_true).
        If B_support is in the equivalence class of the structure generating the
        precision matrix prec_true, then the minimum kld should be theoretically zero.

        Args:
            obj_fn: function to minimize (input = Q)
        """
        assert B_support.shape[0] == B_support.shape[1]
        p = B_support.shape[0]

        self.approx_flops += B_support.shape[0] ** 3

        def _obj(y: np.ndarray):
            Q = y_to_Q(y)
            v, gQ = self.obj_fn(Q)
            gy = grad_Q_to_grad_y(gQ)

            return v, gy

        # make bounds based on supp matrix
        Q_support = (B_support != 0) | (np.eye(p) != 0)
        bounds = [
            (None, None) if Q_support[i, j] else (0, 0)
            for i in range(p)
            for j in range(p)
        ]

        best_obj = np.inf
        Q_best = None
        wait = 0
        n_iters = 0
        while (
            wait <= self.patience
            and best_obj > theo_best + self.tol
            and n_iters <= self.max_iters
        ):
            if self.init_generator is None:
                x0 = np.random.randn(p**2)
            else:
                x0 = self.init_generator()
            res = sopt.minimize(_obj, x0, jac=True, method="L-BFGS-B", bounds=bounds)
            # print(n_iters, wait, res.fun, np.sum(np.abs(res.x) > 1e-2))
            if res.fun < best_obj - self.tol:
                best_obj = res.fun
                Q_best = y_to_Q(res.x).copy()
                wait = 0
            else:
                wait += 1
            n_iters += 1

        self.call_count += 1

        return best_obj, Q_best
