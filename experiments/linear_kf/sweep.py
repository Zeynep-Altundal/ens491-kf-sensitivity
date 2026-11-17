"""One-parameter-at-a-time Monte Carlo sweep of filter mis-tuning.

For each sweep exactly one filter quantity is scaled by ``s`` while all other
quantities are kept at their true values:

* ``Q``:  Q_filter = s * Q_true
* ``R``:  R_filter = s * R_true
* ``P0``: P0_filter = s * P0_true

In every sweep the true initial state is drawn from N(x0_true, P0_true) and the
filter is initialised at x0_true. The same simulated trajectories are reused
for every setting (common random numbers).

Usage (from the repository root)::

    python -m experiments.linear_kf.sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .kf import kalman_filter
from .model import DEFAULT_STEPS, LinearGaussianModel, constant_velocity_model, simulate

S_GRID = np.logspace(-3, 3, 13)
N_TRAJ = 200
DEFAULT_SEED = 0
SWEEPS = ("Q", "R", "P0")
METRICS = ("pos_rmse", "vel_rmse", "mean_nees", "mean_nis")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def filter_matrices(model: LinearGaussianModel, sweep: str, s: float) -> dict:
    """Filter parameters with one quantity scaled by ``s``."""
    params = dict(F=model.F, H=model.H, Q=model.Q, R=model.R, x0=model.x0, P0=model.P0)
    if sweep not in SWEEPS:
        raise ValueError(f"unknown sweep {sweep!r}; expected one of {SWEEPS}")
    params[sweep] = s * params[sweep]
    return params


def evaluate(x_true: np.ndarray, y: np.ndarray, k_min: int = 1, **filter_params) -> dict:
    """Run the filter and compute error and consistency metrics.

    Metrics are averaged over all trajectories and over the steps
    k = k_min, ..., T (default: all steps). NEES uses the filter's posterior
    covariance and NIS the filter's innovation covariance.
    """
    res = kalman_filter(y, **filter_params)
    sl = slice(k_min - 1, None)  # result index j corresponds to step k = j + 1
    err = (res.estimates - x_true[..., 1:, :])[..., sl, :]
    nu = res.innovations[..., sl, :]
    nees = np.einsum("...ti,tij,...tj->...t", err, np.linalg.inv(res.covariances[sl]), err)
    nis = np.einsum(
        "...ti,tij,...tj->...t", nu, np.linalg.inv(res.innovation_covariances[sl]), nu
    )
    return {
        "pos_rmse": float(np.sqrt(np.mean(err[..., 0] ** 2))),
        "vel_rmse": float(np.sqrt(np.mean(err[..., 1] ** 2))),
        "mean_nees": float(np.mean(nees)),
        "mean_nis": float(np.mean(nis)),
    }


def run_sweeps(
    model: LinearGaussianModel,
    steps: int = DEFAULT_STEPS,
    n_traj: int = N_TRAJ,
    seed: int = DEFAULT_SEED,
    s_grid: np.ndarray = S_GRID,
) -> dict[str, np.ndarray]:
    """Run all sweeps; returns ``{sweep: array (len(s_grid), 1 + len(METRICS))}``."""
    x_true, y = simulate(model, steps, n_traj, np.random.default_rng(seed))
    results = {}
    for sweep in SWEEPS:
        rows = []
        for s in s_grid:
            metrics = evaluate(x_true, y, **filter_matrices(model, sweep, s))
            rows.append([s] + [metrics[name] for name in METRICS])
        results[sweep] = np.array(rows)
    return results


def save_results(results: dict[str, np.ndarray], out_dir: Path = RESULTS_DIR) -> list[Path]:
    """Write one CSV per sweep; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sweep, table in results.items():
        path = out_dir / f"sweep_{sweep.lower()}.csv"
        np.savetxt(path, table, delimiter=",", header=",".join(("s",) + METRICS),
                   comments="", fmt="%.10g")
        paths.append(path)
    return paths


def load_results(out_dir: Path = RESULTS_DIR) -> dict[str, np.ndarray]:
    """Read the CSVs written by :func:`save_results`."""
    return {
        sweep: np.loadtxt(out_dir / f"sweep_{sweep.lower()}.csv", delimiter=",", skiprows=1)
        for sweep in SWEEPS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-traj", type=int, default=N_TRAJ)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()

    results = run_sweeps(constant_velocity_model(), args.steps, args.n_traj, args.seed)
    for path in save_results(results):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
