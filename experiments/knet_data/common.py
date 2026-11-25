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
