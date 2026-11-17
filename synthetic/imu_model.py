"""IMU measurement model.

The ideal IMU follows directly from the ground truth:

    gyro_clean  = w_b
    accel_clean = f_b = R_wb^T (a_w - g_w)

The corrupted measurement of each sensor is

    y[k] = clean[k] + b[k] + n[k] + o[k]

with, each independently switchable,

* white noise ``n[k] ~ N(0, sigma_d^2)`` per axis, ``sigma_d = density / sqrt(dt)``;
* bias ``b[k] = b_turn_on + r[k]``, where ``b_turn_on`` is a constant turn-on
  bias and ``r`` is a random walk ``r[0] = 0``,
  ``r[k+1] = r[k] + sigma_bw * sqrt(dt) * n``, ``n ~ N(0, 1)``;
* outlier spikes ``o[k]``: per sample and per axis, with probability
  ``outlier_prob``, a spike of size ``outlier_magnitude`` with random sign.

Units: gyro noise density [rad/s/sqrt(Hz)], gyro bias random walk
[rad/s^2/sqrt(Hz)]; accelerometer noise density [m/s^2/sqrt(Hz)],
accelerometer bias random walk [m/s^3/sqrt(Hz)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation import quat_to_rotmat
from .trajectories import TrajectoryState

GRAVITY_W = np.array([0.0, 0.0, -9.81])


def ideal_imu(state: TrajectoryState) -> tuple[np.ndarray, np.ndarray]:
    """Error-free gyro and accelerometer signals, each of shape (N, 3)."""
    r_wb = quat_to_rotmat(state.q)
    accel = np.einsum("nji,nj->ni", r_wb, state.a - GRAVITY_W)  # R_wb^T (a_w - g_w)
    return state.w_b.copy(), accel


@dataclass(frozen=True)
class SensorErrorModel:
    """Error parameters of one three-axis sensor (gyro or accelerometer)."""

    noise_density: float = 0.0
    turn_on_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_rw: float = 0.0
    outlier_prob: float = 0.0
    outlier_magnitude: float = 0.0
    enable_white_noise: bool = True
    enable_turn_on_bias: bool = True
    enable_bias_rw: bool = True
    enable_outliers: bool = True


@dataclass(frozen=True)
class CorruptedSensor:
    measured: np.ndarray  # (N, 3)
    bias: np.ndarray  # (N, 3) true bias b[k]
    outlier_mask: np.ndarray  # (N, 3) bool, True where a spike was added


def corrupt_sensor(
    clean: np.ndarray,
    dt: float,
    model: SensorErrorModel,
    rng_noise: np.random.Generator,
    rng_bias_rw: np.random.Generator,
    rng_outliers: np.random.Generator,
) -> CorruptedSensor:
    """Apply the enabled error terms to a clean (N, 3) signal sampled at ``dt``.

    Each error source draws from its own generator, so enabling or disabling one
    source does not change the realisation of the others.
    """
    clean = np.asarray(clean, dtype=float)
    n = clean.shape[0]

    bias = np.zeros_like(clean)
    if model.enable_turn_on_bias:
        bias += np.asarray(model.turn_on_bias, dtype=float)
    if model.enable_bias_rw and model.bias_rw > 0.0:
        steps = model.bias_rw * np.sqrt(dt) * rng_bias_rw.standard_normal((n - 1, 3))
        bias[1:] += np.cumsum(steps, axis=0)

    noise = np.zeros_like(clean)
    if model.enable_white_noise and model.noise_density > 0.0:
        noise = (model.noise_density / np.sqrt(dt)) * rng_noise.standard_normal((n, 3))

    mask = np.zeros(clean.shape, dtype=bool)
    spikes = np.zeros_like(clean)
    if model.enable_outliers and model.outlier_prob > 0.0:
        mask = rng_outliers.random((n, 3)) < model.outlier_prob
        signs = 2.0 * rng_outliers.integers(0, 2, size=(n, 3)) - 1.0
        spikes = np.where(mask, signs * model.outlier_magnitude, 0.0)

    return CorruptedSensor(measured=clean + bias + noise + spikes, bias=bias, outlier_mask=mask)
