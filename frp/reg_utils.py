import numpy as np
from typing import Optional

# TODO: Check if pseudo_sign is necessary. Try using np.sign instead.


def pseudo_sign(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, -1)


def soft_thrs(x: np.ndarray, thrs: float):
    return np.sign(x) * np.maximum(np.abs(x) - thrs, 0)


class RegManager:
    def __init__(
        self,
        lam: Optional[float] = None,
        lr: Optional[float] = None,
        slope: Optional[float] = None,
        thrs: Optional[float] = None,
        gamma: Optional[float] = None,
    ):
        self.lam = lam
        self.lr = lr
        self.slope = slope
        self.thrs = thrs
        self.gamma = gamma

    def l1(self):
        assert self.lam is not None
        assert self.lam > 0

        def reg_l1(W: np.ndarray):
            return self.lam * np.sum(np.abs(W)), self.lam * np.sign(W)

        return reg_l1

    def tanh(self):
        assert self.lam is not None
        assert self.lam > 0
        assert self.slope is not None
        assert self.slope > 0
        assert self.thrs is not None
        h_max = 1
        h_min = np.tanh(-self.slope * self.thrs)

        def reg_tanh(W: np.ndarray):
            h_raw = np.tanh(self.slope * (np.abs(W) - self.thrs))
            h = np.sum(h_raw - h_min) / (h_max - h_min)
            grad_h = self.slope * (1 - h_raw**2) / (h_max - h_min) * np.sign(W)
            # grad_h > 0 at W = 0
            return self.lam * h, self.lam * grad_h

        return reg_tanh

    def scad(self):
        assert self.lam is not None
        assert self.lam > 0
        assert self.gamma is not None
        assert self.gamma > 2

        def reg_scad(W: np.ndarray):
            absW = np.abs(W)
            h = np.where(
                absW < self.lam,
                self.lam * absW,
                np.where(
                    absW < self.gamma * self.lam,
                    (2 * self.gamma * self.lam * absW - absW**2 - self.lam**2)
                    / (2 * (self.gamma - 1)),
                    (self.lam**2) * (self.gamma + 1) / 2,
                ),
            )
            grad_h = np.sign(W) * np.where(
                absW < self.lam,
                self.lam,
                np.where(
                    absW < self.gamma * self.lam,
                    (self.gamma * self.lam - absW) / (self.gamma - 1),
                    0,
                ),
            )
            return np.sum(h), grad_h

        return reg_scad

    def mcp(self):
        assert self.lam is not None
        assert self.lam > 0
        assert self.gamma is not None
        assert self.gamma > 1

        def reg_mcp(W: np.ndarray):
            absW = np.abs(W)
            h = np.where(
                absW < self.lam * self.gamma,
                self.lam * absW - (absW**2) / (2 * self.gamma),
                self.gamma * (self.lam**2) / 2,
            )
            grad_h = np.sign(W) * np.where(
                absW < self.lam * self.gamma, (self.lam - absW / self.gamma), 0
            )
            return np.sum(h), grad_h

        return reg_mcp

    def zero(self):
        def reg_zero(W: np.ndarray):
            return 0, np.zeros_like(W)

        return reg_zero

    def l1_thrs(self):
        assert self.lam is not None
        assert self.lam > 0
        assert self.lr is not None
        assert self.lr > 0

        def l1_thrs_inner(W: np.ndarray):
            thrs = self.lam * self.lr
            return np.sign(W) * np.maximum(np.abs(W) - thrs, 0)

        return l1_thrs_inner

    def mcp_thrs(self):
        assert self.lam is not None
        assert self.lam > 0
        assert self.gamma is not None
        assert self.gamma > 1
        assert self.lr is not None
        assert self.lr > 0

        def mcp_thrs_inner(W: np.ndarray):
            absW = np.abs(W)
            thrs = self.lam * self.lr
            s1 = soft_thrs(W, thrs)
            return np.where(
                absW < self.lam * self.gamma, (self.gamma / (self.gamma - 1)) * s1, W
            )

        return mcp_thrs_inner

    def scad_thrs(self):
        assert self.lam is not None
        assert self.lam > 0
        assert self.gamma is not None
        assert self.gamma > 1
        assert self.lr is not None
        assert self.lr > 0

        def scad_thrs_inner(W: np.ndarray):
            absW = np.abs(W)
            thrs = self.lam * self.lr
            s1 = soft_thrs(W, thrs)
            s2 = soft_thrs(W, (self.gamma / (self.gamma - 1)) * thrs)
            return np.where(
                absW < 2 * self.lam,
                s1,
                np.where(
                    absW < self.gamma * self.lam,
                    ((self.gamma - 1) / (self.gamma - 2)) * s2,
                    W,
                ),
            )

        return scad_thrs_inner
