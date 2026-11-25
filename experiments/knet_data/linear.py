"""Linear benchmark of the KalmanNet paper: data generation and a Kalman filter check.

What this script does
---------------------
1. Builds the 2x2 linear Gaussian model of the KalmanNet paper (arXiv:2107.10043, Sec. IV-B):

       x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q),   Q = q^2 I
       y_t = H x_t     + v_t,   v_t ~ N(0, R),   R = r^2 I,   nu = q^2 / r^2

   F is in controllable canonical form (for 2x2: [[1, 1], [0, 1]]) and H = I.
2. Generates train / validation / test trajectories for four scenarios (see SCENARIOS) and saves
   each one in the common format of ``common.py`` under ``DATA_DIR``.
3. Runs a Kalman filter on the test split as a sanity check and writes ``results/linear_check.csv``:
   - "KF true model": the filter knows the true F and H. In a linear Gaussian model this is the
     optimal (MMSE) filter, so its empirical MSE must match the MSE predicted by its own
     covariance P. A mismatch would mean the generator and the filter disagree.
   - "KF given model": the filter uses F_model and H_model. Under mismatch this is the baseline
     that KalmanNet should beat.
   - "Measurement only": x estimated from y alone (H_model^+ y), the "noise floor".

Values that the paper does not state (x0, split sizes, r^2) follow the reference KalmanNet code
(``Simulations/Linear_canonical/parameters.py`` and ``main_linear_canonical.py``, commit 828a2cf).
The model is written here from the equations; no reference source file is copied.

Usage (from the repository root)::

    python -m experiments.knet_data.linear
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat
from experiments.linear_kf.kf import kalman_filter

from .common import (
    DATA_DIR, N_TEST, N_TRAIN, N_VAL, load_dataset, mse_db, per_trajectory_mse, rotation,
    save_dataset, split_rngs,
)

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "linear_check.csv"

SEED = 0
R2 = 1.0                   # observation noise variance r^2, as in the reference code
X0 = np.zeros(2)           # known initial state (reference code: m1_0 = 0, m2_0 = 0)
F0 = np.array([[1.0, 1.0],  # controllable canonical form, 2x2
               [0.0, 1.0]])
H0 = np.eye(2)

# name: (T, nu [dB], rotation of F in the data [deg], rotation of H in the data [deg])
# The filters are always given F0 and H0; a non-zero angle therefore creates a model mismatch.
SCENARIOS = {
    "full":   (100, -20.0, 0.0, 0.0),   # full information, reference-code default
    "rotF10": (20, 0.0, 10.0, 0.0),     # state-evolution mismatch, paper Fig. 6a
    "rotF20": (20, 0.0, 20.0, 0.0),     # state-evolution mismatch, paper Fig. 6a
    "rotH10": (100, -20.0, 0.0, 10.0),  # observation mismatch (~5 % misalignment), paper Fig. 6b
}


def scenario_matrices(name: str) -> dict:
    """True and model matrices plus noise covariances of one scenario."""
    T, nu_db, rot_f, rot_h = SCENARIOS[name]
    q2 = 10 ** (nu_db / 10.0) * R2
    return {
        "T": T, "nu_db": nu_db, "q2": q2,
        # Paper eq. (16): F_alpha = Rot(alpha) F0 generates the data, the filters get F0.
        # (The reference code multiplies on the right and, in architecture-#1, gives the rotated
        # matrix to the filter instead; we follow the paper.) H0 = I, so the side does not matter.
        "F_true": rotation(rot_f) @ F0, "H_true": rotation(rot_h) @ H0,
        "F_model": F0.copy(), "H_model": H0.copy(),
        "Q": q2 * np.eye(2), "R": R2 * np.eye(2),
    }


def simulate(F, H, Q, R, x0, n_traj: int, T: int, rng: np.random.Generator):
    """Simulate ``n_traj`` trajectories of length T. Returns x (N, T, n) and y (N, T, m).

    The first stored state is x_1 = F x0 + w_1; x0 itself is not part of the trajectory.
    """
    n, m = F.shape[0], H.shape[0]
    w = rng.multivariate_normal(np.zeros(n), Q, size=(n_traj, T))
    v = rng.multivariate_normal(np.zeros(m), R, size=(n_traj, T))
    x = np.empty((n_traj, T, n))
    x_prev = np.broadcast_to(x0, (n_traj, n))
    for t in range(T):
        x_prev = x_prev @ F.T + w[:, t]
        x[:, t] = x_prev
    y = x @ H.T + v
    return x, y


def generate(name: str, seed: int = SEED) -> Path:
    """Generate and save one scenario. Returns the path of the ``.npz`` file."""
    mats = scenario_matrices(name)
    rngs = split_rngs(seed)
    arrays = {k: mats[k] for k in ("F_true", "H_true", "F_model", "H_model", "Q", "R")}
    arrays["x0"] = X0
    for split, n_traj in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
        x, y = simulate(mats["F_true"], mats["H_true"], mats["Q"], mats["R"], X0,
                        n_traj, mats["T"], rngs[split])
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x, y
    meta = {
        "benchmark": "linear", "scenario": name, "seed": seed, "T": mats["T"],
        "nu_db": mats["nu_db"], "r2": R2, "inv_r2_db": float(-10 * np.log10(R2)) + 0.0,  # + 0.0 avoids "-0.0"
        "rot_F_deg": SCENARIOS[name][2], "rot_H_deg": SCENARIOS[name][3],
        "source": "KalmanNet paper Sec. IV-B; unstated values from reference code 828a2cf",
    }
    return save_dataset(DATA_DIR / f"linear_{name}.npz", arrays, meta)


def run_kf(arrays: dict, which: str):
    """Kalman filter on the test split with the 'true' or the 'model' matrices."""
    return kalman_filter(arrays["y_test"], arrays[f"F_{which}"], arrays[f"H_{which}"],
                         arrays["Q"], arrays["R"], arrays["x0"], np.zeros((2, 2)))


def check(name: str) -> list[dict]:
    """Sanity check of one saved scenario. Returns one CSV row per method."""
    arrays, meta = load_dataset(DATA_DIR / f"linear_{name}.npz")
    x = arrays["x_test"]
    true_res = run_kf(arrays, "true")
    estimates = {
        "KF true model": true_res.estimates,
        "KF given model": run_kf(arrays, "model").estimates,
        "Measurement only": arrays["y_test"] @ np.linalg.pinv(arrays["H_model"]).T,
    }
    # MSE that the optimal filter predicts for itself: average of trace(P_t) / n over time.
    predicted = 10 * np.log10(np.mean(np.trace(true_res.covariances, axis1=1, axis2=2)) / x.shape[2])
    rows = []
    for method, x_hat in estimates.items():
        per_traj = per_trajectory_mse(x_hat, x)
        _, lo, hi = bootstrap(mean_stat, per_traj)
        rows.append({
            "scenario": name, "T": meta["T"], "nu_db": meta["nu_db"], "inv_r2_db": meta["inv_r2_db"],
            "method": method, "mse_db": round(mse_db(x_hat, x), 3),
            "ci_lo_db": round(10 * np.log10(lo), 3), "ci_hi_db": round(10 * np.log10(hi), 3),
            "predicted_db": round(float(predicted), 3) if method == "KF true model" else "",
        })
    return rows


def main() -> None:
    rows = []
    for name in SCENARIOS:
        path = generate(name)
        print(f"saved {path}")
        rows += check(name)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'scenario':8s} {'method':18s} {'MSE [dB]':>9s} {'95% CI':>18s} {'predicted':>10s}")
    for r in rows:
        ci = f"[{r['ci_lo_db']:.2f}, {r['ci_hi_db']:.2f}]"
        print(f"{r['scenario']:8s} {r['method']:18s} {r['mse_db']:9.2f} {ci:>18s} {str(r['predicted_db']):>10s}")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
