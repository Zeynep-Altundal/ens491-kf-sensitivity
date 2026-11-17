"""Standard linear Kalman filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanFilterResult:
    """Filter outputs for k = 1, ..., T.

    ``estimates`` and ``innovations`` carry the batch dimensions of the input
    measurements. Covariances do not depend on the measurements and are
    therefore shared across the batch.
    """

    estimates: np.ndarray  # (..., T, n) posterior means
    covariances: np.ndarray  # (T, n, n) posterior covariances
    innovations: np.ndarray  # (..., T, m) y[k] - H x_pred[k]
    innovation_covariances: np.ndarray  # (T, m, m) H P_pred H^T + R


def kalman_filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> KalmanFilterResult:
    """Run a Kalman filter on measurements ``y`` of shape (..., T, m).

    The filter starts from the prior ``(x0, P0)`` at k = 0 and performs a
    prediction followed by an update for each measurement. ``x0`` may be a
    single vector or broadcastable to the batch shape ``(..., n)``. The
    covariance update uses the Joseph form for numerical robustness.
    """
    y = np.asarray(y, dtype=float)
    F, H, Q, R, P0 = (np.asarray(a, dtype=float) for a in (F, H, Q, R, P0))
    steps, m = y.shape[-2], y.shape[-1]
    n = F.shape[0]
    batch = y.shape[:-2]
    eye = np.eye(n)

    x = np.broadcast_to(np.asarray(x0, dtype=float), batch + (n,)).copy()
    P = P0.copy()

    estimates = np.empty(batch + (steps, n))
    covariances = np.empty((steps, n, n))
    innovations = np.empty(batch + (steps, m))
    innovation_covariances = np.empty((steps, m, m))

    for k in range(steps):
        # Prediction.
        x = x @ F.T
        P = F @ P @ F.T + Q
        # Update.
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T  # P H^T S^-1, using symmetry of S and P
        nu = y[..., k, :] - x @ H.T
        x = x + nu @ K.T
        A = eye - K @ H
        P = A @ P @ A.T + K @ R @ K.T

        estimates[..., k, :] = x
        covariances[k] = P
        innovations[..., k, :] = nu
        innovation_covariances[k] = S

    return KalmanFilterResult(estimates, covariances, innovations, innovation_covariances)
