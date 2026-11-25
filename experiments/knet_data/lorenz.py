"""Lorenz-attractor benchmark of the KalmanNet paper in the common dataset format, with an EKF check.

What this script does
---------------------
The Lorenz model itself is the existing implementation in ``experiments/filter_family/lorenz.py``
(paper eqs. (18)-(21): dt = 0.02, Taylor order J = 5 for the data, nu = -20 dB, x0 = [1, 1, 1]).
This script only exports it to the common format of ``common.py`` so that KalmanNet and the
classical filters use the same trajectories, and adds the observation-rotation case of the paper.

1. Two scenarios, one ``.npz`` file per noise level 1/r^2:
   - "full" (paper Table V / VII): y = x + v. Training and validation T = 100, test T = 2000.
   - "rotH1" (paper Table VIII): y = H_rot x + v, H_rot = Rz Ry Rx with 1 deg about each axis
     (the reference code's definition); filters are given H = I. Training / validation T = 100,
     test T = 1000.
2. EKF check on the test split, with a batched EKF (``common.batched_ekf``; same equations as the
   filter-family EKF, all trajectories at once). Exact Jacobian by central differences, as in
   ``filter_family/lorenz.py``. Results go to ``results/lorenz_check.csv``:
   - full:  EKF with J = 5 (Table V) and with J = 2 (process mismatch, Table VII); for J = 2 also
     with the process noise Q tuned on the validation split, as in ``sinusoidal.py`` (with a wrong
     model and the true Q the EKF trusts its model too much at high 1/r^2)
   - rotH1: EKF with the true H_rot (reference, Table VIII "theta = 0") and with H = I (theta = 1 deg)

Usage (from the repository root)::

    python -m experiments.knet_data.lorenz
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from experiments.filter_family.lorenz import DT, FD_STEP, J_DATA, J_MISMATCH, NU_DB, P0_SCALE, X0, make_f

from .common import (
    DATA_DIR, N_TEST, N_TRAIN, N_VAL, batched_ekf, load_dataset, mse_db, per_trajectory_mse,
    save_dataset, split_rngs,
)

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "lorenz_check.csv"
SEED = 0
T_TRAIN = 100

# name: (test length, 1/r^2 levels [dB], rotation of H in the data [deg per axis])
SCENARIOS = {
    "full":  (2000, (0.0, 10.0, 20.0, 30.0, 40.0), 0.0),
    "rotH1": (1000, (0.0, 10.0, 20.0, 30.0), 1.0),
}

# EKF MSE [dB] in the paper, keyed by (table, 1/r^2 [dB]).
PAPER = {
    "V":         dict(zip((0.0, 10.0, 20.0, 30.0, 40.0), (-10.45, -20.37, -30.40, -40.39, -49.89))),
    "VII":       dict(zip((10.0, 20.0, 30.0, 40.0), (-19.47, -23.63, -33.51, -41.15))),
    "VIII th=0": dict(zip((0.0, 10.0, 20.0, 30.0), (-10.40, -20.41, -30.50, -40.45))),
    "VIII th=1": dict(zip((0.0, 10.0, 20.0, 30.0), (-9.80, -16.50, -18.19, -18.57))),
}


def rotation_xyz(deg: float) -> np.ndarray:
    """Rz Ry Rx with the same angle about each axis (reference code, Lorenz parameters)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ ry @ rx


def noise_levels(inv_r2_db: float) -> tuple[float, float]:
    r2 = 10 ** (-inv_r2_db / 10.0)
    return 10 ** (NU_DB / 10.0) * r2, r2


def simulate(H, q2, r2, n_traj, T, rng):
    """x_t = f(x_{t-1}) + w_t (J = 5), y_t = H x_t + v_t, from the fixed x0. Shapes (N, T, 3)."""
    f = make_f(J_DATA)
    w = rng.standard_normal((n_traj, T, 3)) * math.sqrt(q2)
    v = rng.standard_normal((n_traj, T, 3)) * math.sqrt(r2)
    x = np.empty((n_traj, T, 3))
    prev = np.broadcast_to(X0, (n_traj, 3))
    for t in range(T):
        prev = f(prev) + w[:, t]
        x[:, t] = prev
    return x, x @ H.T + v


def file_name(scenario: str, inv_r2_db: float) -> Path:
    return DATA_DIR / f"lorenz_{scenario}_inv_r2_{inv_r2_db:+06.2f}dB.npz"


def generate(scenario: str, inv_r2_db: float, seed: int = SEED) -> Path:
    T_test, _, rot = SCENARIOS[scenario]
    q2, r2 = noise_levels(inv_r2_db)
    H_true = rotation_xyz(rot)
    rngs = split_rngs(seed)
    arrays = {"x0": X0, "Q": q2 * np.eye(3), "R": r2 * np.eye(3), "H_true": H_true, "H_model": np.eye(3)}
    for split, n_traj, T in (("train", N_TRAIN, T_TRAIN), ("val", N_VAL, T_TRAIN), ("test", N_TEST, T_test)):
        arrays[f"x_{split}"], arrays[f"y_{split}"] = simulate(H_true, q2, r2, n_traj, T, rngs[split])
    meta = {
        "benchmark": "lorenz", "scenario": scenario, "seed": seed, "dt": DT, "J_data": J_DATA,
        "T_train": T_TRAIN, "T_test": T_test, "nu_db": NU_DB, "inv_r2_db": inv_r2_db,
        "rot_H_deg_per_axis": rot,
        "source": "KalmanNet paper eqs. (18)-(21), Tables V, VII, VIII; model from filter_family/lorenz.py",
    }
    return save_dataset(file_name(scenario, inv_r2_db), arrays, meta)


def jacobian(f):
    """Batched Jacobian of f by central differences, shape (N, 3, 3)."""
    def jac(x):
        cols = [(f(x + FD_STEP * e) - f(x - FD_STEP * e)) / (2 * FD_STEP) for e in np.eye(3)]
        return np.stack(cols, axis=-1)
    return jac


# Process-noise multipliers tried by the grid search (filter Q = multiplier * true Q).
Q_GRID = tuple(10.0 ** k for k in range(0, 7))


def run_ekf(arrays: dict, J: int, H: np.ndarray, split: str = "test", q_mult: float = 1.0) -> np.ndarray:
    """Batched EKF on one split with Taylor order J, observation matrix H and Q scaled by q_mult."""
    f = make_f(J)
    N = arrays[f"y_{split}"].shape[0]
    H_b = np.broadcast_to(H, (N, 3, 3))
    with np.errstate(all="ignore"):   # a badly tuned filter may diverge; such runs are counted
        x_hat, _ = batched_ekf(arrays[f"y_{split}"], f, jacobian(f), lambda x: x @ H.T, lambda x: H_b,
                               q_mult * arrays["Q"], arrays["R"], arrays["x0"], P0_SCALE * np.eye(3))
    return x_hat


def tune_q(arrays: dict, J: int, H: np.ndarray) -> float:
    """Q multiplier with the lowest validation MSE (the test split is not used)."""
    def score(k):
        x_hat = run_ekf(arrays, J, H, "val", k)
        return mse_db(x_hat, arrays["x_val"]) if np.isfinite(x_hat).all() else np.inf
    scores = {k: score(k) for k in Q_GRID}
    return min(scores, key=scores.get)


def check(scenario: str, inv_r2_db: float) -> list[dict]:
    arrays, _ = load_dataset(file_name(scenario, inv_r2_db))
    if scenario == "full":
        runs = (("EKF J=5", J_DATA, np.eye(3), "V"), ("EKF J=2 (process mismatch)", J_MISMATCH, np.eye(3), "VII"),
                ("EKF J=2, tuned Q", J_MISMATCH, np.eye(3), "VII"))
    else:
        runs = (("EKF true H", J_DATA, arrays["H_true"], "VIII th=0"),
                ("EKF H=I (1 deg mismatch)", J_DATA, np.eye(3), "VIII th=1"))
    rows = []
    for label, J, H, table in runs:
        q_mult = tune_q(arrays, J, H) if "tuned" in label else 1.0
        x_hat = run_ekf(arrays, J, H, "test", q_mult)
        ok = np.isfinite(x_hat).all(axis=(1, 2))   # drop diverged trajectories, count them
        x_t = arrays["x_test"][ok]
        per_traj_db = 10 * np.log10(per_trajectory_mse(x_hat[ok], x_t))
        rows.append({
            "scenario": scenario, "inv_r2_db": inv_r2_db, "method": label, "q_mult": q_mult,
            "mse_db": round(mse_db(x_hat[ok], x_t), 2), "std_db": round(float(np.std(per_traj_db)), 2),
            "diverged": int((~ok).sum()), "paper_db": PAPER[table].get(inv_r2_db, ""), "paper_table": table,
        })
    return rows


def main() -> None:
    rows = []
    for scenario, (_, levels, _) in SCENARIOS.items():
        for d in levels:
            print(f"saved {generate(scenario, d)}")
            rows += check(scenario, d)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'scenario':8s} {'1/r2':>6s} {'method':28s} {'Q x':>6s} {'ours [dB]':>10s} {'std':>5s} {'paper':>8s} {'table':>10s}")
    for r in rows:
        paper = f"{r['paper_db']:.2f}" if r["paper_db"] != "" else "-"
        print(f"{r['scenario']:8s} {r['inv_r2_db']:6.1f} {r['method']:28s} {r['q_mult']:6.0e} {r['mse_db']:10.2f} "
              f"{r['std_db']:5.2f} {paper:>8s} {r['paper_table']:>10s}")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()