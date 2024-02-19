from typing import Callable, Tuple
import numpy as np
from utils import check_numpy


def loss_wrapper_W_delta(loss_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]]):
    """
    loss_fn: takes in prec. matrix and returns (loss, grad_W, grad_delta)
    """

    def loss_fn_W_delta(
        W: np.ndarray, delta: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        p = W.shape[0]
        A = np.eye(p) - W
        loss, g_loss = loss_fn(A @ np.diag(delta) @ A.T)
        g_W = -2 * g_loss @ A @ np.diag(delta)
        g_delta = np.array(
            [A[:, k].reshape(1, -1) @ g_loss @ A[:, k].reshape(-1, 1) for k in range(p)]
        ).reshape(-1)
        return loss, g_W, g_delta

    return loss_fn_W_delta


def loss_wrapper_Q(loss_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]]):
    """
    wraps Q -> QQ^T
    loss_fn: takes in prec. matrix and returns (loss, grad_Q)
    """

    def loss_fn_Q(Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        loss, g_loss = loss_fn(Q @ Q.T)
        g_Q = 2 * g_loss @ Q

        return loss, g_Q

    return loss_fn_Q


def wrapper_x_Q(func: Callable[[np.ndarray], Tuple[float, np.ndarray]]):
    def func_x(y: np.ndarray):
        Q = x_to_Q(y)
        return func(Q)

    return func_x


def Q_to_x(Q: np.ndarray):
    check_numpy([Q])
    flatQ = Q.reshape(-1)
    x_pos = np.where(flatQ >= 0, flatQ, 0)
    x_neg = np.where(flatQ < 0, -flatQ, 0)

    return np.concatenate([x_pos, x_neg])


def x_to_Q(x: np.ndarray):
    check_numpy([x])
    p = int(np.sqrt(len(x) / 2))
    x_pos, x_neg = x[: p**2], x[p**2 :]

    return (x_pos - x_neg).reshape(p, p)


def x_to_absQ(x: np.ndarray):
    check_numpy([x])
    p = int(np.sqrt(len(x) / 2))
    x_pos, x_neg = x[: p**2], x[p**2 :]

    return (x_pos + x_neg).reshape(p, p)


def W_delta_to_x(W: np.ndarray, delta: np.ndarray):
    check_numpy([W, delta])
    x_pos = np.where(W >= 0, W, 0)
    x_neg = np.where(W < 0, -W, 0)

    return np.concatenate([x_pos.reshape(-1), x_neg.reshape(-1), delta])


def x_to_W_delta(x: np.ndarray):
    p = int(np.round((-1 + np.sqrt(1 + 8 * len(x))) / 4))
    W = (x[: p**2] - x[p**2 : 2 * p**2]).reshape(p, p)
    delta = x[2 * p**2 :]

    return W, delta


def grad_W_delta_to_grad_x(grad_W: np.ndarray, grad_delta: np.ndarray):
    return np.concatenate([grad_W.reshape(-1), -grad_W.reshape(-1), grad_delta])


def grad_Q_to_grad_x(grad_Q: np.ndarray):
    return np.concatenate([grad_Q.reshape(-1), -grad_Q.reshape(-1)])


def grad_absQ_to_grad_x(grad_Q: np.ndarray):
    return np.concatenate([grad_Q.reshape(-1), grad_Q.reshape(-1)])


def wrapper_x_W_delta(
    func: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray, np.ndarray]]
):
    def func_x(x: np.ndarray):
        W, delta = x_to_W_delta(x)
        return func(W, delta)

    return func_x


def Q_to_y(Q: np.ndarray):
    check_numpy([Q])
    y = Q.reshape(-1)
    return y


def y_to_Q(y: np.ndarray):
    check_numpy([y])
    p = int(np.sqrt(len(y)))
    Q = y.reshape(p, p)

    return Q


def wrapper_y_Q(func: Callable[[np.ndarray], Tuple[float, np.ndarray]]):
    def func_y(y: np.ndarray):
        Q = y_to_Q(y)
        return func(Q)

    return func_y


def grad_Q_to_grad_y(grad_Q: np.ndarray):
    return grad_Q.reshape(-1)
