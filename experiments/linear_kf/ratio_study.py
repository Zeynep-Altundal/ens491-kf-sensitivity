"""Ratio versus absolute-scale study for the linear Kalman filter.

Background: scaling all filter covariances (Q_f, R_f, P0_f) by a common factor
c leaves every Kalman gain, and therefore every estimate and the RMSE,
unchanged, while NEES and NIS are multiplied by 1/c. The estimates thus depend
only on the ratios Q_f/R_f and P0_f/R_f. This module provides the experiments
that check this and explain the Q x 0.001 versus R x 1000 discrepancy of the
one-at-a-time sweeps:

1. Open point: Q x 0.001, R x 1000 and R x 1000 with P0 x 1000 (and the mirrored
   Q x 1000 / R x 0.001 cases) on the full T = 200 run and on a steady-state
   window k >= T/2 of a T = 2000 run.
2. Gain convergence: the number of steps until K_k is within 1 % of the
   steady-state gain.
3. A 13 x 13 grid over (Q scale, R scale) with P0 at truth, for both windows,
   plus the spread of each metric along lines of constant Q_f/R_f.

Usage (from the repository root)::

    python -m experiments.linear_kf.ratio_study
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .kf import kalman_filter
from .model import DEFAULT_STEPS, LinearGaussianModel, constant_velocity_model, simulate
from .sweep import DEFAULT_SEED, METRICS, N_TRAJ, RESULTS_DIR, S_GRID, evaluate

LONG_STEPS = 2000
GAIN_REL_TOL = 0.01

# Each configuration scales the true (Q, R, P0) by the given factors.
CONFIGS: dict[str, tuple[float, float, float]] = {
    "truth": (1.0, 1.0, 1.0),
    "Q x 0.001": (1e-3, 1.0, 1.0),
    "R x 1000": (1.0, 1e3, 1.0),
    "R x 1000, P0 x 1000": (1.0, 1e3, 1e3),
    "Q x 1000": (1e3, 1.0, 1.0),
    "R x 0.001": (1.0, 1e-3, 1.0),
    "R x 0.001, P0 x 0.001": (1.0, 1e-3, 1e-3),
}

# (label, steps, k_min): full short run and steady-state window of a long run.
WINDOWS = (
    ("T200_full", DEFAULT_STEPS, 1),
    ("T2000_steady", LONG_STEPS, LONG_STEPS // 2),
)


def scaled_params(model: LinearGaussianModel, sq: float, sr: float, sp0: float) -> dict:
    """Filter parameters with Q, R and P0 scaled by the given factors."""
    return dict(F=model.F, H=model.H, Q=sq * model.Q, R=sr * model.R, x0=model.x0,
                P0=sp0 * model.P0)


def steady_state_gain(params: dict, tol: float = 1e-14, max_iter: int = 10**7) -> tuple:
    """Iterate the Riccati recursion until P converges; returns (K, P, iterations)."""
    F, H, Q, R = params["F"], params["H"], params["Q"], params["R"]
    P = np.array(params["P0"], dtype=float)
    n = F.shape[0]
    for it in range(1, max_iter + 1):
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = np.linalg.solve(S, H @ Pp).T
        A = np.eye(n) - K @ H
        P_new = A @ Pp @ A.T + K @ R @ K.T
        if np.max(np.abs(P_new - P) / np.abs(P_new)) < tol:
            return K, P_new, it
        P = P_new
    raise RuntimeError("Riccati recursion did not converge")


def gain_sequence(params: dict, steps: int) -> np.ndarray:
    """Kalman gains K_k for k = 1, ..., steps (data independent), shape (steps, n, m)."""
    dummy = np.zeros((steps, params["H"].shape[0]))
    return kalman_filter(dummy, **params).gains


def convergence_step(gains: np.ndarray, k_ss: np.ndarray, rel_tol: float = GAIN_REL_TOL):
    """First step k after which every gain entry stays within rel_tol of k_ss; None if never."""
    rel = np.max(np.abs(gains - k_ss) / np.abs(k_ss), axis=(1, 2))
    outside = np.nonzero(rel >= rel_tol)[0]
    if outside.size == 0:
        return 1
    last = outside[-1] + 1  # convert index to step k
    return None if last >= len(gains) else last + 1


def run_open_point(model: LinearGaussianModel, data: dict) -> list[dict]:
    rows = []
    for name, (sq, sr, sp0) in CONFIGS.items():
        params = scaled_params(model, sq, sr, sp0)
        k_ss, _, _ = steady_state_gain(params)
        k_conv = convergence_step(gain_sequence(params, LONG_STEPS), k_ss)
        row = {"config": name, "q_scale": sq, "r_scale": sr, "p0_scale": sp0,
               "gain_conv_step": "" if k_conv is None else k_conv}
        for label, steps, k_min in WINDOWS:
            x_true, y = data[label]
            metrics = evaluate(x_true, y, k_min=k_min, **params)
            for key in METRICS:
                row[f"{key}_{label}"] = metrics[key]
        rows.append(row)
    return rows


def run_grid(model: LinearGaussianModel, data: dict, label: str, k_min: int) -> list[dict]:
    x_true, y = data[label]
    rows = []
    for sq in S_GRID:
        for sr in S_GRID:
            metrics = evaluate(x_true, y, k_min=k_min, **scaled_params(model, sq, sr, 1.0))
            rows.append({"q_scale": sq, "r_scale": sr, **metrics})
    return rows


def diagonal_spread(grid_rows: list[dict]) -> list[dict]:
    """Spread of each metric along lines of constant Q_f/R_f (P0 fixed at truth)."""
    groups: dict[float, list[dict]] = {}
    for row in grid_rows:
        key = round(np.log10(row["q_scale"]) - np.log10(row["r_scale"]), 6)
        groups.setdefault(key, []).append(row)
    out = []
    for key in sorted(groups):
        rows = groups[key]
        entry = {"log10_q_over_r": key, "n_points": len(rows)}
        for metric in METRICS:
            vals = np.array([r[metric] for r in rows])
            entry[f"{metric}_min"] = vals.min()
            entry[f"{metric}_max"] = vals.max()
        out.append(entry)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})


def simulate_windows(model: LinearGaussianModel, seed: int = DEFAULT_SEED) -> dict:
    """One dataset per window, each from a fresh generator with the same seed."""
    return {label: simulate(model, steps, N_TRAJ, np.random.default_rng(seed))
            for label, steps, _ in WINDOWS}


def main() -> None:
    model = constant_velocity_model()
    data = simulate_windows(model)

    open_rows = run_open_point(model, data)
    write_csv(open_rows, RESULTS_DIR / "open_point.csv")
    print("wrote", RESULTS_DIR / "open_point.csv")

    for label, _, k_min in WINDOWS:
        grid_rows = run_grid(model, data, label, k_min)
        write_csv(grid_rows, RESULTS_DIR / f"grid_{label}.csv")
        write_csv(diagonal_spread(grid_rows), RESULTS_DIR / f"grid_diagonals_{label}.csv")
        print("wrote", RESULTS_DIR / f"grid_{label}.csv")
        print("wrote", RESULTS_DIR / f"grid_diagonals_{label}.csv")


if __name__ == "__main__":
    main()
