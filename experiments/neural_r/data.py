"""Training pattern generator, validation runs and test sets (``README.md``, Part 1).

All sets use the measurement-noise simulator of ``experiments/filter_family/time_varying_r``
(imported unchanged): ``R_true(k) = c(k) R_nom``.
"""

from __future__ import annotations

import numpy as np

from experiments.filter_family.models import linear_cv_model
from experiments.filter_family.time_varying_r import noise_scale, simulate_tv

STEPS = 400
N_RUNS = 100
BATCH = 32
TRAIN_SEED0 = 1100
VAL_SEED = 1000
VAL_BURST_SEED = 1001
VAL_PATTERNS = ("constant", "step", "ramp", "burst")
PATTERN_MIX = (("burst", 0.4), ("step", 0.2), ("ramp", 0.2), ("constant", 0.2))


def model():
    return linear_cv_model()


def _train_scale(rng: np.random.Generator) -> np.ndarray:
    T = STEPS
    k = np.arange(1, T + 1)
    u = rng.random()
    cum = np.cumsum([p for _, p in PATTERN_MIX])
    pattern = PATTERN_MIX[int(np.searchsorted(cum, u, side="right"))][0]
    if pattern == "constant":
        return np.ones(T)
    if pattern == "step":
        f = float(rng.choice([2.0, 5.0, 10.0]))
        k0 = int(rng.integers(100, 301))
        level = f if rng.random() < 0.5 else 1.0 / f
        return np.where(k <= k0, 1.0, level)
    if pattern == "ramp":
        f = float(rng.choice([2.0, 5.0, 10.0]))
        r = 1.0 + (f - 1.0) * (k - 1) / (T - 1)
        return r if rng.random() < 0.5 else r[::-1].copy()
    f = float(rng.choice([10.0, 25.0]))
    L = int(rng.choice([5, 10, 20]))
    n_b = int(rng.integers(2, 7))
    c = np.ones(T)
    starts: list[int] = []
    while len(starts) < n_b:
        s = int(rng.integers(0, T - L + 1))
        if all(s + L <= t or t + L <= s for t in starts):
            starts.append(s)
    for s in starts:
        c[s:s + L] = f
    return c


def training_batch(it: int):
    """Batch of iteration ``it``: (c, x, y) with c (BATCH, T); seed TRAIN_SEED0 + it."""
    rng = np.random.default_rng(TRAIN_SEED0 + it)
    c = np.stack([_train_scale(rng) for _ in range(BATCH)])
    x, y = simulate_tv(model(), c, rng)
    return c, x, y


def validation_set(pattern: str):
    """Validation runs of the time-varying benchmark (seed 1000, burst seed 1001)."""
    c = noise_scale(pattern, steps=STEPS, n_runs=N_RUNS, seed=VAL_BURST_SEED)
    x, y = simulate_tv(model(), c, np.random.default_rng(VAL_SEED))
    return c, x, y


def _block_bursts(factor: float, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = STEPS // 4
    c = np.ones((N_RUNS, STEPS))
    for i in range(N_RUNS):
        for b in range(4):
            s = b * block + int(rng.integers(0, block - length + 1))
            c[i, s:s + length] = factor
    return c


def _tile(v: np.ndarray) -> np.ndarray:
    return np.tile(v, (N_RUNS, 1))


TEST_SETS = ("constant", "step", "ramp", "burst",
             "ho_step", "ho_ramp", "ho_burst_short", "ho_burst_long", "ho_sine")
BURST_SETS = ("burst", "ho_burst_short", "ho_burst_long")
HELD_OUT_BURSTS = ("ho_burst_short", "ho_burst_long")


def test_set(name: str):
    """(c, x, y) of a test set as defined in the README."""
    k = np.arange(1, STEPS + 1)
    if name in ("constant", "step", "ramp", "burst"):
        c, seed = noise_scale(name, steps=STEPS, n_runs=N_RUNS, seed=1), 0
    elif name == "ho_step":
        c, seed = _tile(np.where(k <= 60, 1.0, np.where(k <= 340, 20.0, 2.0))), 2000
    elif name == "ho_ramp":
        c, seed = _tile(1.0 + 29.0 * (k - 1) / (STEPS - 1)), 2001
    elif name == "ho_burst_short":
        c, seed = _block_bursts(50.0, 3, 2052), 2002
    elif name == "ho_burst_long":
        c, seed = _block_bursts(25.0, 40, 2053), 2003
    elif name == "ho_sine":
        c, seed = _tile(10.0 ** (0.5 + 0.5 * np.sin(2.0 * np.pi * k / 100.0))), 2004
    else:
        raise ValueError(name)
    x, y = simulate_tv(model(), c, np.random.default_rng(seed))
    return c, x, y
