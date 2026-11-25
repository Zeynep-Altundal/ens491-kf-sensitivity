"""Shared helpers for the KalmanNet benchmark datasets (linear, sinusoidal, Lorenz, NCLT).

Every dataset is stored as one ``.npz`` file with the same keys, so that the classical filters and
KalmanNet read exactly the same trajectories:

    x_train, y_train, x_val, y_val, x_test, y_test   true states (N, T, n) and measurements (N, T, m)
    x0                                               initial state (n,), the same for every trajectory
    F_true, H_true                                   matrices used to generate the data (linear models)
    F_model, H_model                                 matrices given to the filters (differ under mismatch)
    Q, R                                             noise covariances used to generate the data
    meta                                             JSON string: scenario name, seed, T, nu, 1/r^2, ...

The time axis comes before the state axis, as in ``experiments/linear_kf/kf.py``. The reference
KalmanNet code uses (N, n, T); the KalmanNet adapter transposes when it loads a file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Datasets are not committed; they live next to the other raw data (~/data/nclt, ~/data/euroc).
# Set the environment variable KNET_DATA_DIR to store them somewhere else.
DATA_DIR = Path(os.environ.get("KNET_DATA_DIR", Path.home() / "data" / "knet_benchmarks"))

# Split sizes of the reference KalmanNet code (training, validation, test trajectories).
N_TRAIN, N_VAL, N_TEST = 1000, 100, 200
SPLITS = ("train", "val", "test")


def split_rngs(seed: int) -> dict[str, np.random.Generator]:
    """One independent random generator per split, all derived from a single seed.

    Using separate streams means that changing N_TRAIN does not change the test trajectories.
    """
    children = np.random.SeedSequence(seed).spawn(len(SPLITS))
    return {name: np.random.default_rng(child) for name, child in zip(SPLITS, children)}


def rotation(deg: float) -> np.ndarray:
    """2-D rotation matrix for an angle in degrees."""
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def mse_db(x_hat: np.ndarray, x: np.ndarray) -> float:
    """Mean squared error over trajectories, time steps and state components, in dB."""
    return float(10.0 * np.log10(np.mean((x_hat - x) ** 2)))


def per_trajectory_mse(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """MSE of each trajectory (linear scale), shape (N,). Used for bootstrap intervals."""
    return np.mean((x_hat - x) ** 2, axis=(1, 2))


def save_dataset(path: Path, data: dict[str, np.ndarray], meta: dict) -> Path:
    """Write a dataset in the common format. ``data`` holds the array keys listed above."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta, sort_keys=True), **data)
    return path


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a dataset written by ``save_dataset``. Returns ``(arrays, meta)``."""
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if k != "meta"}
        meta = json.loads(str(f["meta"]))
    return arrays, meta




def batched_ekf(y, f, F_jac, h, H_jac, Q, R, x0, P0):
    """Extended Kalman filter run on many trajectories at once (vectorised over the batch).

    Same equations as ``experiments/filter_family/filters.py::EKF`` (predict, then update with
    the Joseph-form covariance), but every step processes all N trajectories together, which is
    much faster in numpy. ``f``, ``h`` take (N, n); ``F_jac``, ``H_jac`` return (N, n, n) / (N, m, n).
    ``y``: measurements (N, T, m). Returns posterior means (N, T, n) and covariances (N, T, n, n).
    """
    N, T, m = y.shape
    n = Q.shape[0]
    x = np.broadcast_to(np.asarray(x0, dtype=float), (N, n)).copy()
    P = np.broadcast_to(np.asarray(P0, dtype=float), (N, n, n)).copy()
    I = np.eye(n)
    xs, Ps = np.empty((N, T, n)), np.empty((N, T, n, n))
    for t in range(T):
        # predict: P <- F P F^T + Q, with F the Jacobian at the previous estimate
        Fj = F_jac(x)
        x = f(x)
        P = Fj @ P @ Fj.transpose(0, 2, 1) + Q
        # update: S = H P H^T + R, K = P H^T S^-1, Joseph form for P
        Hj = H_jac(x)
        S = Hj @ P @ Hj.transpose(0, 2, 1) + R
        K = np.linalg.solve(S, Hj @ P).transpose(0, 2, 1)
        nu = y[:, t] - h(x)
        x = x + np.einsum("nij,nj->ni", K, nu)
        A = I - K @ Hj
        P = A @ P @ A.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)
        xs[:, t], Ps[:, t] = x, P
    return xs, Ps




def batched_eskf(y, f, F_jac, h, H_jac, Q, R, x0, P0):
    """Error-state Kalman filter for a plain vector state, run on many trajectories at once.

    Written in error-state form on purpose, to compare it with ``batched_ekf``:
      - the nominal state x_nom is propagated with the full non-linear f (no noise),
      - the filter estimates the error dx = x_true - x_nom, with covariance P,
      - after an update the error is injected into the nominal state and reset to zero.
    For a vector state the injection is a plain addition and the reset Jacobian is the identity,
    so every step is algebraically the same as the EKF. The two differ only when the state lives
    on a manifold (e.g. a quaternion attitude, as in ``experiments/eskf_sensitivity``), where the
    injection is x_nom (+) dx and the reset Jacobian is not the identity.
    """
    N, T, m = y.shape
    n = Q.shape[0]
    x_nom = np.broadcast_to(np.asarray(x0, dtype=float), (N, n)).copy()
    P = np.broadcast_to(np.asarray(P0, dtype=float), (N, n, n)).copy()
    I = np.eye(n)
    G = I                                   # reset Jacobian: identity for a vector state
    xs, Ps = np.empty((N, T, n)), np.empty((N, T, n, n))
    for t in range(T):
        # 1) propagate the nominal state and the error covariance (the error mean stays zero)
        Fj = F_jac(x_nom)
        x_nom = f(x_nom)
        P = Fj @ P @ Fj.transpose(0, 2, 1) + Q
        # 2) estimate the error from the measurement residual
        Hj = H_jac(x_nom)
        S = Hj @ P @ Hj.transpose(0, 2, 1) + R
        K = np.linalg.solve(S, Hj @ P).transpose(0, 2, 1)
        dx = np.einsum("nij,nj->ni", K, y[:, t] - h(x_nom))
        A = I - K @ Hj
        P = A @ P @ A.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)
        # 3) inject the error into the nominal state, then reset the error to zero
        x_nom = x_nom + dx
        P = G @ P @ G.T
        xs[:, t], Ps[:, t] = x_nom, P
    return xs, Ps