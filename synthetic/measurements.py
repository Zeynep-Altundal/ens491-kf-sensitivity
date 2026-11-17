"""Low-rate position measurements.

    z[j] = p_true(t_j) + v[j] + o[j],   v[j] ~ N(0, diag(sigma^2))

* Dropout: measurements with ``start <= t_j < start + duration`` for any
  configured window are marked unavailable and set to NaN.
* Outliers: per epoch, with probability ``outlier_prob``, an offset of norm
  ``outlier_magnitude`` in a uniformly random 3D direction is added.

Random numbers are drawn for every epoch before dropout is applied, so the
dropout configuration does not change the noise or outlier realisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionMeasurementModel:
    sigma: tuple[float, float, float] = (0.0, 0.0, 0.0)  # [m] per axis
    dropout_windows: tuple[tuple[float, float], ...] = ()  # (start [s], duration [s])
    outlier_prob: float = 0.0
    outlier_magnitude: float = 0.0  # [m]
    enable_noise: bool = True
    enable_dropout: bool = True
    enable_outliers: bool = True


def dropout_mask(t: np.ndarray, windows) -> np.ndarray:
    """True where ``t`` lies inside any half-open window ``[start, start + duration)``."""
    t = np.asarray(t, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    for start, duration in windows:
        mask |= (t >= start) & (t < start + duration)
    return mask


def position_measurements(
    t: np.ndarray,
    p_true: np.ndarray,
    model: PositionMeasurementModel,
    rng_noise: np.random.Generator,
    rng_outliers: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate measurements at times ``t`` (M,) from true positions (M, 3)."""
    p_true = np.asarray(p_true, dtype=float)
    m = p_true.shape[0]
    z = p_true.copy()

    if model.enable_noise:
        z += rng_noise.standard_normal((m, 3)) * np.asarray(model.sigma, dtype=float)

    outlier = np.zeros(m, dtype=bool)
    if model.enable_outliers and model.outlier_prob > 0.0:
        outlier = rng_outliers.random(m) < model.outlier_prob
        direction = rng_outliers.standard_normal((m, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        z += np.where(outlier[:, None], model.outlier_magnitude * direction, 0.0)

    available = np.ones(m, dtype=bool)
    if model.enable_dropout:
        available = ~dropout_mask(t, model.dropout_windows)
    z[~available] = np.nan

    return {"z": z, "available": available, "outlier_mask": outlier & available}
