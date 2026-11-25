"""ESKF = EKF on the four KalmanNet benchmarks (vector states).


---------------------
In all four benchmarks the state is a plain vector (no attitude quaternion). For such a state the
error-state Kalman filter (ESKF) injects the estimated error by addition and its reset Jacobian is
the identity, so it is algebraically the same filter as the EKF (and, for a linear model, the KF).
This script shows it numerically: on the test split of each dataset it runs

    - ``common.batched_ekf``   (EKF)
    - ``common.batched_eskf``  (ESKF written in error-state form: nominal state, error, inject, reset)
    - ``experiments/linear_kf/kf.py`` (KF) where the model is linear (linear, NCLT)

with the same model, noise covariances and initial state, and reports the largest difference of
the estimates and the MSE of each filter (``results/eskf_equivalence.csv``). Differences at the
level of floating-point rounding (about 1e-12 relative) mean "identical".

The datasets must exist (run linear.py, sinusoidal.py, lorenz.py and nclt.py first).
The repository's ``experiments/eskf_sensitivity/eskf.py`` is an IMU/quaternion ESKF and cannot run
on these models; the ESKF here is the generic vector-state version.

Usage (from the repository root)::

    python -m experiments.knet_data.eskf_equivalence
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.filter_family.lorenz import J_DATA, P0_SCALE, make_f
from experiments.linear_kf.kf import kalman_filter

from . import lorenz, nclt, sinusoidal
from .common import DATA_DIR, batched_ekf, batched_eskf, load_dataset, mse_db

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "eskf_equivalence.csv"


def linear_model(F, H):
    """f, F_jac, h, H_jac for a linear model, batched over (N, n)."""
    def F_jac(x):
        return np.broadcast_to(F, (x.shape[0],) + F.shape)

    def H_jac(x):
        return np.broadcast_to(H, (x.shape[0],) + H.shape)

    return (lambda x: x @ F.T), F_jac, (lambda x: x @ H.T), H_jac


def sinusoidal_model(p):
    """f, F_jac, h, H_jac of the sinusoidal benchmark; Jacobians are diagonal (per component)."""
    def diag(d):
        J = np.zeros(d.shape + (d.shape[-1],))
        idx = np.arange(d.shape[-1])
        J[..., idx, idx] = d
        return J

    F_jac = lambda x: diag(p["alpha"] * p["beta"] * np.cos(p["beta"] * x + p["phi"]))
    H_jac = lambda x: diag(2.0 * p["a"] * p["b"] * (p["b"] * x + p["c"]))
    return (lambda x: sinusoidal.f(x, p)), F_jac, (lambda x: sinusoidal.h(x, p)), H_jac


def nclt_noise() -> tuple[float, float]:
    """q^2, r^2 from the NCLT check if it has been run, otherwise 1 and 1."""
    if nclt.RESULTS_CSV.exists():
        with nclt.RESULTS_CSV.open() as fh:
            row = next(csv.DictReader(fh))
        return float(row["q2"]), float(row["r2"])
    return 1.0, 1.0


def cases():
    """(dataset name, arrays, model functions, Q, R, x0, P0, is_linear) for the four benchmarks."""
    a, _ = load_dataset(DATA_DIR / "linear_full.npz")
    yield ("linear (full)", a, linear_model(a["F_model"], a["H_model"]), a["Q"], a["R"],
           a["x0"], np.zeros((2, 2)), True)

    a, _ = load_dataset(sinusoidal.file_name(0.0))
    yield ("sinusoidal (1/r2 = 0 dB)", a, sinusoidal_model(sinusoidal.PARAMS["full"]), a["Q"], a["R"],
           a["x0"], np.zeros((2, 2)), False)

    a, _ = load_dataset(lorenz.file_name("full", 10.0))
    f = make_f(J_DATA)
    funcs = (f, lorenz.jacobian(f)) + linear_model(np.eye(3), np.eye(3))[2:]
    yield ("lorenz (full, 1/r2 = 10 dB)", a, funcs, a["Q"], a["R"], a["x0"], P0_SCALE * np.eye(3), False)

    a, _ = load_dataset(DATA_DIR / f"nclt_{nclt.SESSION}.npz")
    q2, r2 = nclt_noise()
    yield ("nclt (test window)", a, linear_model(nclt.F, nclt.H), q2 * nclt.Q_STRUCT, r2 * np.eye(2),
           a["x0_test"], np.zeros((4, 4)), True)


def compare(name, a, funcs, Q, R, x0, P0, is_linear) -> dict:
    y, x = a["y_test"], a["x_test"]
    x_ekf, P_ekf = batched_ekf(y, *funcs, Q, R, x0, P0)
    x_eskf, P_eskf = batched_eskf(y, *funcs, Q, R, x0, P0)
    row = {
        "dataset": name, "n_traj": y.shape[0], "T": y.shape[1],
        "max_abs_diff_eskf_ekf": float(np.max(np.abs(x_eskf - x_ekf))),
        "max_abs_diff_P_eskf_ekf": float(np.max(np.abs(P_eskf - P_ekf))),
        "mse_db_ekf": round(mse_db(x_ekf, x), 4), "mse_db_eskf": round(mse_db(x_eskf, x), 4),
        "max_abs_diff_kf_ekf": "", "mse_db_kf": "",
    }
    if is_linear:
        F_lin, H_lin = funcs[1](x[:, 0])[0], funcs[3](x[:, 0])[0]
        x0_b = np.broadcast_to(x0, (y.shape[0], Q.shape[0]))
        x_kf = np.stack([kalman_filter(yy, F_lin, H_lin, Q, R, xx0, P0).estimates for yy, xx0 in zip(y, x0_b)])
        row["max_abs_diff_kf_ekf"] = float(np.max(np.abs(x_kf - x_ekf)))
        row["mse_db_kf"] = round(mse_db(x_kf, x), 4)
    return row


def main() -> None:
    rows = [compare(*case) for case in cases()]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'dataset':30s} {'|ESKF-EKF| max':>15s} {'|KF-EKF| max':>13s} {'EKF [dB]':>9s} {'ESKF [dB]':>10s} {'KF [dB]':>8s}")
    for r in rows:
        kf_diff = f"{r['max_abs_diff_kf_ekf']:.1e}" if r["max_abs_diff_kf_ekf"] != "" else "-"
        kf_mse = f"{r['mse_db_kf']:.2f}" if r["mse_db_kf"] != "" else "-"
        print(f"{r['dataset']:30s} {r['max_abs_diff_eskf_ekf']:15.1e} {kf_diff:>13s} "
              f"{r['mse_db_ekf']:9.2f} {r['mse_db_eskf']:10.2f} {kf_mse:>8s}")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()