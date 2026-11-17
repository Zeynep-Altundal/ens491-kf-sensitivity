"""Kalman filter family behind one interface.

Every filter is constructed from a :class:`~.models.StateSpaceModel`, an
initial mean and covariance, and optionally filter values of Q and R that
differ from the model's true values. ``predict()`` returns the predicted
``(x, P)``; ``update(z)`` returns an :class:`UpdateResult` with the posterior
estimate and covariance, the innovation and the innovation covariance.
Innovations are formed with the model's residual function, so angular
components (the bearing) are wrapped to (-pi, pi].

Members: :class:`KF`, :class:`EKF`, :class:`IEKF`, :class:`UKF`,
:class:`AdaptiveEKF` (adaptive estimation of R; on a linear model the EKF is
the KF, so this is also the adaptive KF).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from experiments.linear_kf.kf import kf_predict, kf_update

from .models import StateSpaceModel


@dataclass(frozen=True)
class UpdateResult:
    x: np.ndarray
    P: np.ndarray
    innovation: np.ndarray
    innovation_cov: np.ndarray


class Filter:
    """Common base: holds the model, the current mean and covariance and the filter Q, R."""

    name = "filter"

    def __init__(self, model: StateSpaceModel, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray | None = None, R: np.ndarray | None = None):
        self.model = model
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(model.Q if Q is None else Q, dtype=float)
        self.R = np.array(model.R if R is None else R, dtype=float)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update(self, z: np.ndarray) -> UpdateResult:
        raise NotImplementedError


class KF(Filter):
    """Linear Kalman filter; uses the predict/update steps of ``experiments/linear_kf/kf.py``.

    Requires a linear model: ``F_jacobian`` and ``H_jacobian`` are evaluated
    once and treated as the constant matrices F and H.
    """

    name = "KF"

    def __init__(self, model, x0, P0, Q=None, R=None):
        super().__init__(model, x0, P0, Q, R)
        self.F = model.F_jacobian(self.x)
        self.H = model.H_jacobian(self.x)

    def predict(self):
        self.x, self.P = kf_predict(self.x, self.P, self.F, self.Q)
        return self.x, self.P

    def update(self, z):
        nu = z - self.x @ self.H.T
        self.x, self.P, S, _ = kf_update(self.x, self.P, nu, self.H, self.R)
        return UpdateResult(self.x, self.P, nu, S)


class EKF(Filter):
    """Extended Kalman filter (first-order linearisation, Joseph-form update)."""

    name = "EKF"

    def predict(self):
        F = self.model.F_jacobian(self.x)
        self.x = self.model.f(self.x)
        self.P = F @ self.P @ F.T + self.Q
        return self.x, self.P

    def update(self, z):
        H = self.model.H_jacobian(self.x)
        nu = self.model.residual(z, self.model.h(self.x))
        self.x, self.P, S, _ = kf_update(self.x, self.P, nu, H, self.R)
        return UpdateResult(self.x, self.P, nu, S)


class IEKF(EKF):
    """Iterated EKF: Gauss-Newton iterations of the measurement update.

    Starting from the prior ``x_0 = x_pred``, iterate for i = 0, 1, ...::

        H_i  = H(x_i)
        r_i  = residual(z, h(x_i)) - H_i (x_pred - x_i)
        K_i  = P_pred H_i^T (H_i P_pred H_i^T + R)^-1
        x_i+1 = x_pred + K_i r_i

    until ``|x_i+1 - x_i| <= tol`` or ``max_iter`` iterations. The covariance
    uses the Joseph form with the gain and Jacobian of the last iteration.
    The reported innovation and innovation covariance are those of the prior
    (i = 0), which are the quantities a NIS test refers to.
    """

    name = "IEKF"

    def __init__(self, model, x0, P0, Q=None, R=None, max_iter: int = 10, tol: float = 1e-8):
        super().__init__(model, x0, P0, Q, R)
        self.max_iter = max_iter
        self.tol = tol
        self.iterations: list[int] = []

    def update(self, z):
        x_pred, P_pred = self.x, self.P
        x_i = x_pred
        nu0 = S0 = None
        for i in range(self.max_iter):
            H = self.model.H_jacobian(x_i)
            r = self.model.residual(z, self.model.h(x_i)) - H @ (x_pred - x_i)
            x_new, P_new, S, _ = kf_update(x_pred, P_pred, r, H, self.R)
            if i == 0:
                nu0, S0 = r, S
            converged = np.linalg.norm(x_new - x_i) <= self.tol
            x_i = x_new
            if converged:
                break
        self.iterations.append(i + 1)
        self.x, self.P = x_i, P_new
        return UpdateResult(self.x, self.P, nu0, S0)


class UKF(Filter):
    """Unscented Kalman filter with the scaled unscented transform.

    ``lambda = alpha^2 (n + kappa) - n``; sigma points ``x`` and
    ``x +/- sqrt(n + lambda) L_i`` with ``L L^T = P``; weights
    ``Wm_0 = lambda / (n + lambda)``, ``Wc_0 = Wm_0 + 1 - alpha^2 + beta``,
    ``Wm_i = Wc_i = 1 / (2 (n + lambda))``. Defaults ``alpha = 1``,
    ``beta = 2``, ``kappa = 0`` (so ``lambda = 0`` and ``Wm_0 = 0``).
    Sigma points are redrawn from the predicted mean and covariance before the
    update. The measurement mean uses the model's ``measurement_mean``
    (circular mean for the bearing) and all measurement deviations use the
    model's residual.
    """

    name = "UKF"

    def __init__(self, model, x0, P0, Q=None, R=None, alpha: float = 1.0, beta: float = 2.0,
                 kappa: float = 0.0):
        super().__init__(model, x0, P0, Q, R)
        n = model.n
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lam = alpha**2 * (n + kappa) - n
        c = n + self.lam
        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / c
        self.Wc[0] = self.lam / c + 1.0 - alpha**2 + beta
        self.scale = np.sqrt(c)

    def _sigma_points(self, x, P):
        L = np.linalg.cholesky(P) * self.scale
        return np.vstack([x, x + L.T, x - L.T])

    def predict(self):
        X = self.model.f(self._sigma_points(self.x, self.P))
        x = self.Wm @ X
        D = X - x
        self.x = x
        self.P = (D.T * self.Wc) @ D + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x, self.P

    def update(self, z):
        X = self._sigma_points(self.x, self.P)
        Z = self.model.h(X)
        z_hat = self.model.measurement_mean(Z, self.Wm)
        DZ = self.model.residual(Z, z_hat)
        DX = X - self.x
        S = (DZ.T * self.Wc) @ DZ + self.R
        Pxz = (DX.T * self.Wc) @ DZ
        K = np.linalg.solve(S, Pxz.T).T
        nu = self.model.residual(z, z_hat)
        self.x = self.x + K @ nu
        P = self.P - K @ S @ K.T
        self.P = 0.5 * (P + P.T)
        return UpdateResult(self.x, self.P, nu, S)


class AdaptiveEKF(EKF):
    """EKF with residual-based windowed estimation of the measurement covariance R.

    After each update with posterior ``(x+, P+)`` the post-fit residual
    ``e = residual(z, h(x+))`` is stored. Once ``window`` residuals are
    available, the filter value of R used for the next update is::

        R <- (1/W) sum_{j in last W} e_j e_j^T + H(x+) P+ H(x+)^T

    With ``adapt=False`` the filter is exactly the EKF (on a linear model the
    KF). The R used at each update is recorded in ``R_history``.
    """

    name = "AdaptiveEKF"

    def __init__(self, model, x0, P0, Q=None, R=None, window: int = 20, adapt: bool = True):
        super().__init__(model, x0, P0, Q, R)
        self.window = window
        self.adapt = adapt
        self._residuals: deque = deque(maxlen=window)
        self.R_history: list[np.ndarray] = []

    def update(self, z):
        self.R_history.append(self.R.copy())
        res = super().update(z)
        if self.adapt:
            e = self.model.residual(z, self.model.h(self.x))
            self._residuals.append(e)
            if len(self._residuals) == self.window:
                E = np.array(self._residuals)
                H = self.model.H_jacobian(self.x)
                R = E.T @ E / self.window + H @ self.P @ H.T
                self.R = 0.5 * (R + R.T)
        return res


FILTERS = {cls.name: cls for cls in (KF, EKF, IEKF, UKF, AdaptiveEKF)}


def run_filter(filt: Filter, y: np.ndarray) -> dict[str, np.ndarray]:
    """Run ``predict`` and ``update`` for every measurement of one trajectory y (T, m)."""
    steps = y.shape[0]
    n, m = filt.model.n, filt.model.m
    out = {
        "x": np.empty((steps, n)), "P": np.empty((steps, n, n)),
        "nu": np.empty((steps, m)), "S": np.empty((steps, m, m)),
    }
    t0 = time.perf_counter()
    for k in range(steps):
        filt.predict()
        res = filt.update(y[k])
        out["x"][k], out["P"][k], out["nu"][k], out["S"][k] = res.x, res.P, res.innovation, res.innovation_cov
    out["seconds"] = time.perf_counter() - t0
    return out
