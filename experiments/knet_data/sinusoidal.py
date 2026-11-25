"""Synthetic non-linear ("sinusoidal") benchmark of the KalmanNet paper: data and an EKF check.

What this script does
---------------------
1. Builds the toy non-linear model of the KalmanNet paper (arXiv:2107.10043, eq. (17), Table II).
   Both functions act on each of the two state components separately:

       x_t = f(x_{t-1}) + w_t,   f(x) = alpha * sin(beta * x + phi) + delta
       y_t = h(x_t)     + v_t,   h(x) = a * (b * x + c)^2

   "full" parameters generate the data. The "partial" parameters are a simplified model that is
   given to the filters in the mismatch case (Table IV):

       full:    alpha = 0.9, beta = 1.1, phi = 0.1 pi, delta = 0.01, a = 1, b = 1, c = 0
       partial: alpha = 1,   beta = 1,   phi = 0,      delta = 0,    a = 1, b = 1, c = 0

2. For every noise level of the paper's tables (1/r^2 = -12.04, -6.02, 0, 20, 40 dB; nu = -20 dB,
   T = 100) it generates train / validation / test trajectories and saves them in the common
   format of ``common.py``. The data are always generated with the full parameters.
3. Runs an EKF on the test split and compares the MSE with the EKF rows of Tables III and IV of
   the paper (``results/sinusoidal_check.csv``):
   - "full": the EKF knows the full model and the true Q, R (Table III).
   - "partial": the EKF uses the partial model with the true Q, R.
   - "partial, tuned Q": the EKF uses the partial model and a process-noise scale chosen by grid
     search on the validation split. With a wrong f the true Q is far too small (the filter trusts
     its wrong model and ignores the measurements); a larger Q absorbs the model error. The paper
     does not say how its EKF was set up in Table IV, but its numbers are only reachable this way.

Values that the paper does not state (x0 = [0.1, 0.1], P0 = 0, split sizes) follow the reference
code (``architecture-#2``, ``Simulations/Toy_problems/parameters.py``). The reference code uses
delta = 1 and sigma_q = sigma_r = 0.1 (nu = 0 dB); we follow the paper (delta = 0.01, nu = -20 dB).
The model is written here from the equations; no reference source file is copied.

Usage (from the repository root)::

    python -m experiments.knet_data.sinusoidal
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.filter_family.filters import EKF, run_filter
from experiments.filter_family.models import StateSpaceModel

from .common import (
    DATA_DIR, N_TEST, N_TRAIN, N_VAL, load_dataset, mse_db, per_trajectory_mse, save_dataset,
    split_rngs,
)

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "sinusoidal_check.csv"

SEED = 0
T = 100
NU_DB = -20.0
X0 = np.array([0.1, 0.1])
INV_R2_DB = (-12.04, -6.02, 0.0, 20.0, 40.0)   # noise levels of Tables III and IV

PARAMS = {  # paper Table II
    "full":    {"alpha": 0.9, "beta": 1.1, "phi": 0.1 * np.pi, "delta": 0.01, "a": 1.0, "b": 1.0, "c": 0.0},
    "partial": {"alpha": 1.0, "beta": 1.0, "phi": 0.0,         "delta": 0.0,  "a": 1.0, "b": 1.0, "c": 0.0},
}

# EKF MSE [dB] reported in the paper: Table III (filter knows the full model) and
# Table IV (filter uses the partial model), in the order of INV_R2_DB.
PAPER_EKF = {
    "full":    (-6.23, -13.41, -19.58, -39.78, -59.67),
    "partial": (-2.99, -5.07, -7.57, -22.67, -36.55),
}


def f(x, p):
    """State evolution, applied to each component of x (works for (n,) and (N, n))."""
    return p["alpha"] * np.sin(p["beta"] * x + p["phi"]) + p["delta"]


def h(x, p):
    """Observation, applied to each component of x."""
    return p["a"] * (p["b"] * x + p["c"]) ** 2


def noise_levels(inv_r2_db: float) -> tuple[float, float]:
    """(q^2, r^2) for a given 1/r^2 in dB and nu = q^2 / r^2 = NU_DB."""
    r2 = 10 ** (-inv_r2_db / 10.0)
    return 10 ** (NU_DB / 10.0) * r2, r2


def simulate(p: dict, q2: float, r2: float, n_traj: int, rng: np.random.Generator):
    """Simulate ``n_traj`` trajectories. Returns x (N, T, 2) and y (N, T, 2).

    The first stored state is x_1 = f(x0) + w_1; x0 itself is not part of the trajectory.
    """
    w = rng.normal(0.0, np.sqrt(q2), size=(n_traj, T, 2))
    v = rng.normal(0.0, np.sqrt(r2), size=(n_traj, T, 2))
    x = np.empty((n_traj, T, 2))
    x_prev = np.broadcast_to(X0, (n_traj, 2))
    for t in range(T):
        x_prev = f(x_prev, p) + w[:, t]
        x[:, t] = x_prev
    return x, h(x, p) + v


def file_name(inv_r2_db: float) -> Path:
    return DATA_DIR / f"sinusoidal_inv_r2_{inv_r2_db:+06.2f}dB.npz"


def generate(inv_r2_db: float, seed: int = SEED) -> Path:
    """Generate and save the data for one noise level. Returns the path of the ``.npz`` file."""
    q2, r2 = noise_levels(inv_r2_db)
    rngs = split_rngs(seed)
    arrays = {"x0": X0, "Q": q2 * np.eye(2), "R": r2 * np.eye(2)}
    for split, n_traj in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
        arrays[f"x_{split}"], arrays[f"y_{split}"] = simulate(PARAMS["full"], q2, r2, n_traj, rngs[split])
    meta = {
        "benchmark": "sinusoidal", "seed": seed, "T": T, "nu_db": NU_DB, "inv_r2_db": inv_r2_db,
        "q2": q2, "r2": r2, "params_true": PARAMS["full"], "params_model": PARAMS["partial"],
        "source": "KalmanNet paper eq. (17), Table II; unstated values from reference code",
    }
    return save_dataset(file_name(inv_r2_db), arrays, meta)


def ekf_model(p: dict, Q: np.ndarray, R: np.ndarray) -> StateSpaceModel:
    """The model as seen by the EKF. Jacobians are diagonal because f and h act per component."""
    return StateSpaceModel(
        name="sinusoidal",
        f=lambda x: f(x, p),
        F_jacobian=lambda x: np.diag(p["alpha"] * p["beta"] * np.cos(p["beta"] * x + p["phi"])),
        h=lambda x: h(x, p),
        H_jacobian=lambda x: np.diag(2.0 * p["a"] * p["b"] * (p["b"] * x + p["c"])),
        Q=Q, R=R, x0=X0.copy(), P0=np.zeros((2, 2)),
        residual=lambda a, b: a - b, measurement_mean=lambda Z, w: w @ Z, dt=1.0,
    )


# Process-noise multipliers tried by the grid search (filter Q = multiplier * true Q).
Q_GRID = tuple(10.0 ** k for k in range(-2, 9))


def run_ekf(arrays: dict, p: dict, split: str = "test", q_mult: float = 1.0) -> np.ndarray:
    """EKF estimates on one split, shape (N, T, 2). ``q_mult`` scales the filter's Q."""
    model = ekf_model(p, q_mult * arrays["Q"], arrays["R"])
    return np.stack([run_filter(EKF(model, model.x0, model.P0), y)["x"] for y in arrays[f"y_{split}"]])


def tune_q(arrays: dict, p: dict) -> float:
    """Q multiplier with the lowest validation MSE (the test split is not used)."""
    scores = {k: mse_db(run_ekf(arrays, p, "val", k), arrays["x_val"]) for k in Q_GRID}
    return min(scores, key=scores.get)


def check(inv_r2_db: float) -> list[dict]:
    """EKF with the full and with the partial model on one saved noise level."""
    arrays, meta = load_dataset(file_name(inv_r2_db))
    x = arrays["x_test"]
    k_partial = tune_q(arrays, PARAMS["partial"])
    runs = (("full", "full", 1.0), ("partial", "partial", 1.0), ("partial, tuned Q", "partial", k_partial))
    rows = []
    for label, setting, q_mult in runs:
        x_hat = run_ekf(arrays, PARAMS[setting], "test", q_mult)
        per_traj_db = 10 * np.log10(per_trajectory_mse(x_hat, x))
        rows.append({
            "inv_r2_db": inv_r2_db, "filter_model": label, "method": "EKF", "q_mult": q_mult,
            "mse_db": round(mse_db(x_hat, x), 2),
            "std_db": round(float(np.std(per_traj_db)), 2),   # spread over trajectories, like sigma-hat
            "paper_db": PAPER_EKF[setting][INV_R2_DB.index(inv_r2_db)],
            "paper_table": "III" if setting == "full" else "IV",
        })
    return rows


def main() -> None:
    rows = []
    for d in INV_R2_DB:
        print(f"saved {generate(d)}")
        rows += check(d)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'1/r2 [dB]':>9s} {'EKF model':17s} {'Q x':>7s} {'ours [dB]':>10s} {'std':>6s} {'paper [dB]':>11s} {'diff':>6s}")
    for r in rows:
        diff = r["mse_db"] - r["paper_db"]
        print(f"{r['inv_r2_db']:9.2f} {r['filter_model']:17s} {r['q_mult']:7.0e} {r['mse_db']:10.2f} "
              f"{r['std_db']:6.2f} {r['paper_db']:11.2f} {diff:6.2f}")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()