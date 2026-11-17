"""State-space models for the filter-family comparison.

Every model is described by a :class:`StateSpaceModel`:

    x[k+1] = f(x[k]) + w[k],   w ~ N(0, Q)
    y[k]   = h(x[k]) + v[k],   v ~ N(0, R)

with analytic Jacobians, a measurement residual (difference) that handles
angular components, and a weighted measurement mean used by the unscented
transform. ``f`` and ``h`` accept a single state (n,) or a batch (..., n).

Models
------
``linear_cv_model``
    The constant-velocity model of ``experiments/linear_kf`` (reused).
``range_bearing_model``
    2D constant-velocity target, state ``[px, py, vx, vy]``, observed by
    range and bearing from a fixed sensor. The bearing lies in (-pi, pi].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from experiments.linear_kf.model import constant_velocity_model


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    w = np.mod(a + np.pi, 2.0 * np.pi) - np.pi
    return np.where(w == -np.pi, np.pi, w)


@dataclass(frozen=True)
class StateSpaceModel:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    F_jacobian: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    H_jacobian: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray]
    measurement_mean: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dt: float

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def m(self) -> int:
        return self.R.shape[0]


def _linear_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def _weighted_mean(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w @ Z


def linear_cv_model(**kwargs) -> StateSpaceModel:
    """The linear constant-velocity model of ``experiments/linear_kf`` as a StateSpaceModel.

    ``f`` and ``h`` use the same arithmetic as ``experiments/linear_kf/kf.py``
    (``x @ F.T``, ``x @ H.T``), so filters built on this model can reproduce
    the KF bit for bit.
    """
    base = constant_velocity_model(**kwargs)
    F, H = base.F, base.H
    return StateSpaceModel(
        name="linear_cv",
        f=lambda x: x @ F.T,
        F_jacobian=lambda x: F,
        h=lambda x: x @ H.T,
        H_jacobian=lambda x: H,
        Q=base.Q, R=base.R, x0=base.x0, P0=base.P0,
        residual=_linear_residual,
        measurement_mean=_weighted_mean,
        dt=float(F[0, 1]),
    )


def _range_bearing_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d[..., 1] = wrap_angle(d[..., 1])
    return d


def _range_bearing_mean(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean of range and circular weighted mean of bearing."""
    r = w @ Z[:, 0]
    b = np.arctan2(w @ np.sin(Z[:, 1]), w @ np.cos(Z[:, 1]))
    return np.array([r, b])


def range_bearing_model(
    dt: float = 1.0,
    q: float = 0.01,
    sigma_range: float = 0.5,
    sigma_bearing: float = 0.05,
    sensor: tuple[float, float] = (0.0, 0.0),
    x0: tuple[float, float, float, float] = (-50.0, 20.0, 0.5, -1.0),
    p0_diag: tuple[float, float, float, float] = (25.0, 25.0, 1.0, 1.0),
) -> StateSpaceModel:
    """2D constant-velocity target observed by range and bearing.

    State ``[px, py, vx, vy]``; white-noise acceleration with spectral density
    ``q`` on each axis. Measurement ``[range, bearing]`` from ``sensor``,
    bearing ``atan2(py - sy, px - sx)`` in (-pi, pi]. Starting exactly at the
    default mean initial state, the noise-free target crosses the negative x
    axis of the sensor at t = 20 s, where the bearing wraps from +pi to -pi.
    In simulated runs the initial state is drawn from ``N(x0, P0)``, so whether
    and when a run crosses depends on the draw.
    """
    F1 = np.array([[1.0, dt], [0.0, 1.0]])
    Q1 = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    # State ordering [px, py, vx, vy].
    perm = [0, 2, 1, 3]
    F = np.kron(np.eye(2), F1)[np.ix_(perm, perm)]
    Q = np.kron(np.eye(2), Q1)[np.ix_(perm, perm)]
    s = np.asarray(sensor, dtype=float)

    def h(x):
        x = np.asarray(x, dtype=float)
        dx, dy = x[..., 0] - s[0], x[..., 1] - s[1]
        return np.stack([np.hypot(dx, dy), np.arctan2(dy, dx)], axis=-1)

    def H_jacobian(x):
        dx, dy = x[0] - s[0], x[1] - s[1]
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2)
        return np.array([[dx / r, dy / r, 0.0, 0.0],
                         [-dy / r2, dx / r2, 0.0, 0.0]])

    return StateSpaceModel(
        name="range_bearing",
        f=lambda x: x @ F.T,
        F_jacobian=lambda x: F,
        h=h,
        H_jacobian=H_jacobian,
        Q=Q, R=np.diag([sigma_range**2, sigma_bearing**2]),
        x0=np.asarray(x0, dtype=float), P0=np.diag(p0_diag),
        residual=_range_bearing_residual,
        measurement_mean=_range_bearing_mean,
        dt=dt,
    )


def simulate(model: StateSpaceModel, steps: int, n_traj: int,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate trajectories; same draw order and output layout as ``linear_kf.model.simulate``.

    Returns ``x`` (n_traj, steps + 1, n) for k = 0..steps and ``y``
    (n_traj, steps, m) for k = 1..steps. Angular measurement components are
    wrapped by the model's residual convention (the bearing to (-pi, pi]).
    """
    n, m = model.n, model.m
    x0 = model.x0 + rng.standard_normal((n_traj, n)) @ np.linalg.cholesky(model.P0).T
    w = rng.standard_normal((n_traj, steps, n)) @ np.linalg.cholesky(model.Q).T
    v = rng.standard_normal((n_traj, steps, m)) @ np.linalg.cholesky(model.R).T
    x = np.empty((n_traj, steps + 1, n))
    x[:, 0] = x0
    for k in range(steps):
        x[:, k + 1] = model.f(x[:, k]) + w[:, k]
    y = model.h(x[:, 1:]) + v
    y = model.residual(y, np.zeros_like(y))  # wraps angular components
    return x, y
