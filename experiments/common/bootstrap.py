"""Percentile bootstrap over independent Monte Carlo runs.

Conventions are documented in ``experiments/common/README.md``: B = 10 000
resamples, 95 % two-sided percentile interval, resampling generator seeded with
2026, runs as the unit of resampling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

N_BOOT = 10_000
CONF = 0.95
BOOT_SEED = 2026


def mean_stat(x: np.ndarray) -> float:
    return float(np.mean(x))


def rms_stat(mse: np.ndarray) -> float:
    """RMSE from per-run mean squared errors (equal epochs per run)."""
    return float(np.sqrt(np.mean(mse)))


def bootstrap(stat: Callable[..., float], *arrays: np.ndarray, n_boot: int = N_BOOT,
              conf: float = CONF, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Point estimate and percentile interval of ``stat(*arrays)``.

    All arrays share the run axis 0 and are resampled with the same indices,
    so ``stat`` may combine them (paired statistics).
    """
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("all arrays must have the same number of runs")
    est = stat(*arrays)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = np.array([stat(*(a[i] for a in arrays)) for i in idx])
    lo, hi = np.quantile(reps, [(1 - conf) / 2, (1 + conf) / 2])
    return float(est), float(lo), float(hi)


def paired_difference(stat: Callable[[np.ndarray], float], a: np.ndarray, b: np.ndarray,
                      **kwargs) -> tuple[float, float, float]:
    """Interval of ``stat(a) - stat(b)`` with jointly resampled runs."""
    return bootstrap(lambda x, y: stat(x) - stat(y), a, b, **kwargs)


def supported_difference(lo: float, hi: float) -> bool:
    """A paired difference is supported when its interval excludes zero."""
    return lo > 0.0 or hi < 0.0
