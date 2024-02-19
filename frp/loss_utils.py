# All values should be passed as a np.array

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def loss_tanh(c: np.ndarray, w_threshold: np.ndarray):
    def loss_inner(w: np.ndarray):
        l_min = np.tanh(-c * w_threshold)
        l_max = 1
        return (np.tanh(c * (np.abs(w) - w_threshold)) - l_min) / (l_max - l_min)

    return loss_inner


def loss_mcp(gamma: np.ndarray):
    def loss_inner(w: np.ndarray):
        lam = (2 / gamma) ** 0.5
        return np.where(
            np.abs(w) <= gamma * lam, lam * np.abs(w) - w**2 / (2 * gamma), 1
        )

    return loss_inner


def loss_scad(gamma: np.ndarray):
    assert gamma > 1, "gamma must be greater than 1"

    def loss_inner(w: np.ndarray):
        lam = (2 / (gamma + 1)) ** 0.5
        return np.where(
            np.abs(w) < lam,
            lam * np.abs(w),
            np.where(
                np.abs(w) < lam * gamma,
                (2 * gamma * lam * np.abs(w) - w**2 - lam**2) / (2 * (gamma - 1)),
                1,
            ),
        )

    return loss_inner


def sketch_loss_fn(
    loss_fn: Callable[[np.ndarray], np.ndarray], w_post_threshold: np.ndarray
):
    w_range = np.range(0, 2, 0.02)
    losses = [loss_fn(w).item() for w in w_range]
    _, ax = plt.subplots()
    ax.set_title(f"Loss fn")
    ax.plot(w_range, losses, label="loss")
    ax.axvline(x=w_post_threshold, color="r", linestyle="--", label="w_post_threshold")
    ax.legend()
    plt.show()


class LossManager:
    def __init__(self, THETA: np.ndarray):
        self.tg_THETA = THETA
        self.inv_tg_THETA = np.linalg.inv(THETA)
        sgn, logdet = np.linalg.slogdet(THETA)
        self.logdet_tg_THETA = sgn * logdet

    def l2(self):
        def loss_l2(prec: np.ndarray):
            loss = (1 / 2) * np.sum((prec - self.tg_THETA) ** 2)
            g_loss = prec - self.tg_THETA
            return loss, g_loss

        return loss_l2

    # THETA = X.T @ X / n should be given
    def gaussian_lkhd(self):
        def loss_gaussian_lkhd(prec: np.ndarray):
            p = prec.shape[0]
            loss = (
                (p / 2) * np.log(2 * np.pi)
                - (1 / 2) * np.linalg.slogdet(prec)[1]
                + (1 / 2) * np.trace(prec @ self.tg_THETA)
            )
            g_loss = (1 / 2) * (self.tg_THETA - np.linalg.inv(prec))
            return loss, g_loss

        return loss_gaussian_lkhd

    def constant(self):
        def loss_constant(prec: np.ndarray):
            return 0, np.zeros_like(prec)

        return loss_constant

    def kld(self):
        def loss_kld(prec: np.ndarray):
            sgn, logdet = np.linalg.slogdet(prec)
            logdet_prec = sgn * logdet

            loss = (1 / 2) * (
                self.logdet_tg_THETA
                - logdet_prec
                + np.trace(self.inv_tg_THETA @ prec)
                - prec.shape[0]
            )
            g_loss = (1 / 2) * (self.inv_tg_THETA - np.linalg.inv(prec))
            return loss, g_loss

        return loss_kld

    def kld_2nd_approx(self):
        def loss_kld_2nd_approx(prec: np.ndarray):
            THETA_diff = prec - self.tg_THETA
            A = self.inv_tg_THETA @ THETA_diff
            loss = (1 / 4) * np.trace(A @ A)
            g_loss = (1 / 2) * A @ self.inv_tg_THETA
            return loss, g_loss

        return loss_kld_2nd_approx
