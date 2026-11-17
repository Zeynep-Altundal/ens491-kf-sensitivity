"""Linear Gaussian state-space model and Monte Carlo trajectory simulation.

Model
-----
    x[k+1] = F x[k] + w[k],   w[k] ~ N(0, Q_true)
    y[k]   = H x[k] + v[k],   v[k] ~ N(0, R_true)

The initial state is drawn as x[0] ~ N(x0_true, P0_true). Measurements are
available for k = 1, ..., T, i.e. after each propagation step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DT = 0.1
DEFAULT_STEPS = 200


@dataclass(frozen=True)
class LinearGaussianModel:
    """True system matrices and initial-state distribution."""

    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray

    @property
    def n(self) -> int:
        """State dimension."""
        return self.F.shape[0]

    @property
    def m(self) -> int:
        """Measurement dimension."""
        return self.H.shape[0]


def constant_velocity_model(
    dt: float = DEFAULT_DT,
    q: float = 1.0,
    r: float = 1.0,
    x0: tuple[float, float] = (0.0, 1.0),
    p0_diag: tuple[float, float] = (1.0, 1.0),
) -> LinearGaussianModel:
    """Two-state constant-velocity model with position measurements.

    State ``x = [position, velocity]``. The process noise is the discretised
    continuous white-noise acceleration model with spectral density ``q``:
    ``Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]``. The measurement noise
    variance is ``r``.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[r]])
    return LinearGaussianModel(
        F=F, H=H, Q=Q, R=R, x0=np.asarray(x0, dtype=float), P0=np.diag(p0_diag)
    )


def simulate(
    model: LinearGaussianModel,
    steps: int,
    n_traj: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_traj`` independent trajectories.

    Random numbers are drawn in a fixed order (initial states, process noise,
    measurement noise), so a given generator state fully determines the output.

    Returns
    -------
    x : ndarray, shape (n_traj, steps + 1, n)
        True states for k = 0, ..., steps.
    y : ndarray, shape (n_traj, steps, m)
        Measurements for k = 1, ..., steps.
    """
    n, m = model.n, model.m
    x0 = model.x0 + rng.standard_normal((n_traj, n)) @ np.linalg.cholesky(model.P0).T
    w = rng.standard_normal((n_traj, steps, n)) @ np.linalg.cholesky(model.Q).T
    v = rng.standard_normal((n_traj, steps, m)) @ np.linalg.cholesky(model.R).T

    x = np.empty((n_traj, steps + 1, n))
    x[:, 0] = x0
    for k in range(steps):
        x[:, k + 1] = x[:, k] @ model.F.T + w[:, k]
    y = x[:, 1:] @ model.H.T + v
    return x, y
