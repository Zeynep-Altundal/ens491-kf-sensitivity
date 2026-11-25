"""NCLT real-data benchmark of the KalmanNet paper: dataset preparation and a Kalman filter check.

What this script does
---------------------
The KalmanNet paper (arXiv:2107.10043, Sec. IV-E) tracks a Segway robot (UGV) of the Michigan
NCLT dataset, session 2012-01-22, from odometry velocity alone. Each axis is modelled separately
with the linear Wiener velocity model (white-noise acceleration), dt = 1 s:

    state x = [p_x, v_x, p_y, v_y]                (positions in m, velocities in m/s)
    x_t = F x_{t-1} + w_t,   F = blockdiag([[1, dt], [0, 1]], [[1, dt], [0, 1]])
    y_t = H x_t + v_t,       H = [[0, 1, 0, 0], [0, 0, 0, 1]]   (only velocities are measured)

1. Reads the raw files (``NCLT_DIR``, default ``~/data/nclt``):
   - ``groundtruth_2012-01-22.csv``: utime [us], x (north), y (east), z (down) [m], roll, pitch, yaw
   - ``2012-01-22/odometry_mu_100hz.csv``: same columns, wheel odometry relative to the start pose
2. Resamples both to 1 Hz on a common time grid (linear interpolation; NaN rows dropped).
3. Target x: ground-truth positions and velocities (velocity = central difference of position).
   Measurement y: odometry velocity = central difference of odometry position, rotated into the
   ground-truth frame with the ground-truth yaw at the first sample (one-time alignment, like
   initialising a filter). Odometry drifts, so this measurement has real, non-Gaussian errors.
4. Cuts the 1 Hz series into non-overlapping windows in time order: training windows of T = 200,
   then validation windows of T = 200, then one test window of T = 277 (paper: 23 / 2 / 1). If the
   session is too short for 23 training windows, the number of training windows is reduced and
   reported. Each window starts from its own ground-truth state (x0 per window).
5. Saves everything in the common format of ``common.py`` and runs a KF check on the test window,
   with q^2 and r^2 chosen by grid search on the validation windows, as in the paper
   (``results/nclt_check.csv``). Paper, Table X: model-based KF 25.47 dB, KalmanNet 22.2 dB.

Choices the paper does not state (velocity source, alignment, how "unstable readings" were
removed, window order, which error the MSE is computed on) are ours and are documented in the
README. NCLT: Carlevaris-Bianco, Ushani, Eustice, IJRR 2016, Open Database License.

Usage (from the repository root)::

    python -m experiments.knet_data.nclt
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from experiments.linear_kf.kf import kalman_filter

from .common import DATA_DIR, load_dataset, save_dataset

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "nclt_check.csv"
NCLT_DIR = Path(os.environ.get("NCLT_DIR", Path.home() / "data" / "nclt"))
SESSION = "2012-01-22"

DT = 1.0                       # 1 Hz, as in the paper
T_TRAIN, T_VAL, T_TEST = 200, 200, 277
N_TRAIN_WANTED, N_VAL = 23, 2  # paper: 23 training and 2 validation windows, 1 test window

# Wiener velocity model for one axis, and the 4-state model with independent x and y axes.
F_AXIS = np.array([[1.0, DT], [0.0, 1.0]])
Q_AXIS = np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])   # multiplied by q^2
F = np.kron(np.eye(2), F_AXIS)
Q_STRUCT = np.kron(np.eye(2), Q_AXIS)
H = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
POS, VEL = [0, 2], [1, 3]      # indices of positions and velocities in the state

# Grid searched on the validation windows for the KF check.
Q2_GRID = tuple(10.0 ** k for k in np.arange(-6, 4.01, 0.5))
R2_GRID = tuple(10.0 ** k for k in np.arange(-7, 1.01, 0.5))

PAPER_DB = {"KF": 25.47, "KalmanNet": 22.2}   # Table X


def read_csv(path: Path) -> np.ndarray:
    """Read an NCLT CSV (comma separated, may contain '-nan'). Time is converted to seconds."""
    a = np.loadtxt(path, delimiter=",")
    a[:, 0] = a[:, 0] * 1e-6
    return a


def resample(a: np.ndarray, grid: np.ndarray, cols) -> np.ndarray:
    """Linear interpolation of the given columns onto the time grid (rows with NaN are dropped)."""
    a = a[~np.isnan(a).any(axis=1)]
    return np.column_stack([np.interp(grid, a[:, 0], a[:, c]) for c in cols])


def central_diff(p: np.ndarray) -> np.ndarray:
    """Velocity from positions sampled every DT (central difference, one-sided at the ends)."""
    return np.gradient(p, DT, axis=0)


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def build_series(gt: np.ndarray, odom: np.ndarray):
    """1 Hz states x (K, 4) and velocity measurements y (K, 2) on the overlap of both files."""
    gt_ok = gt[~np.isnan(gt).any(axis=1)]
    t0 = max(gt_ok[0, 0], odom[0, 0])
    t1 = min(gt_ok[-1, 0], odom[-1, 0])
    grid = np.arange(t0, t1, DT)
    p_gt = resample(gt, grid, (1, 2))          # x north, y east
    yaw0 = resample(gt, grid[:1], (6,))[0, 0]  # ground-truth heading at the first sample
    p_odom = resample(odom, grid, (1, 2))      # odometry frame, starts at the start pose
    v_gt = central_diff(p_gt)
    v_odom = central_diff(p_odom) @ rot_z(yaw0).T  # rotate odometry velocities into the GT frame
    x = np.column_stack([p_gt[:, 0], v_gt[:, 0], p_gt[:, 1], v_gt[:, 1]])
    return grid, x, v_odom


def split_windows(x: np.ndarray, y: np.ndarray) -> tuple[dict, int]:
    """Non-overlapping windows in time order: train, then validation, then test.

    Returns the split arrays (x_*, y_*, x0_*) and the number of training windows used.
    x0 of a window is the ground-truth state one step before its first sample.
    """
    k = len(x)
    n_train = min(N_TRAIN_WANTED, (k - 1 - N_VAL * T_VAL - T_TEST) // T_TRAIN)
    if n_train < 1:
        raise ValueError(f"series of {k} steps is too short")
    out, start = {}, 1   # start at 1 so that every window has a previous state for x0
    for split, n_win, T in (("train", n_train, T_TRAIN), ("val", N_VAL, T_VAL), ("test", 1, T_TEST)):
        idx = [np.arange(start + i * T, start + (i + 1) * T) for i in range(n_win)]
        out[f"x_{split}"] = np.stack([x[i] for i in idx])
        out[f"y_{split}"] = np.stack([y[i] for i in idx])
        out[f"x0_{split}"] = np.stack([x[i[0] - 1] for i in idx])
        start += n_win * T
    return out, n_train


def generate() -> Path:
    """Read the raw NCLT files, build the windows and save them. Returns the ``.npz`` path."""
    gt = read_csv(NCLT_DIR / f"groundtruth_{SESSION}.csv")
    odom = read_csv(NCLT_DIR / SESSION / "odometry_mu_100hz.csv")
    grid, x, y = build_series(gt, odom)
    arrays, n_train = split_windows(x, y)
    arrays.update({"F_model": F, "H_model": H, "Q_structure": Q_STRUCT, "x0": arrays["x0_test"][0]})
    meta = {
        "benchmark": "nclt", "session": SESSION, "dt": DT, "steps_1hz": int(len(x)),
        "duration_s": float(grid[-1] - grid[0]), "n_train": n_train, "n_val": N_VAL, "n_test": 1,
        "T_train": T_TRAIN, "T_val": T_VAL, "T_test": T_TEST,
        "state": "[p_x (north), v_x, p_y (east), v_y]", "measurement": "odometry velocity [v_x, v_y]",
        "source": "NCLT 2012-01-22 (Carlevaris-Bianco et al., IJRR 2016, ODbL); KalmanNet paper Sec. IV-E",
    }
    return save_dataset(DATA_DIR / f"nclt_{SESSION}.npz", arrays, meta)


def run_kf(arrays: dict, split: str, q2: float, r2: float) -> np.ndarray:
    """KF on all windows of a split; each window starts from its own ground-truth state."""
    est = [kalman_filter(y, F, H, q2 * Q_STRUCT, r2 * np.eye(2), x0, np.zeros((4, 4))).estimates
           for y, x0 in zip(arrays[f"y_{split}"], arrays[f"x0_{split}"])]
    return np.stack(est)


def mse_db_on(x_hat, x, idx) -> float:
    return float(10 * np.log10(np.mean((x_hat[..., idx] - x[..., idx]) ** 2)))


def check() -> list[dict]:
    """Grid-search q^2, r^2 on validation (full-state MSE), then evaluate on the test window."""
    arrays, meta = load_dataset(DATA_DIR / f"nclt_{SESSION}.npz")
    scores = {(q2, r2): mse_db_on(run_kf(arrays, "val", q2, r2), arrays["x_val"], slice(None))
              for q2 in Q2_GRID for r2 in R2_GRID}
    q2, r2 = min(scores, key=scores.get)
    x, x_hat = arrays["x_test"], run_kf(arrays, "test", q2, r2)
    # "Integrate the measured velocity" baseline: position = x0 + cumulative sum of odometry velocity.
    x_int = np.empty_like(x)
    x_int[..., VEL] = arrays["y_test"]
    x_int[..., POS] = arrays["x0_test"][:, None, POS] + np.cumsum(arrays["y_test"], axis=1) * DT
    rows = []
    for method, est in (("KF (tuned q2, r2)", x_hat), ("Integrated odometry velocity", x_int)):
        for part, idx in (("full state", slice(None)), ("position", POS), ("velocity", VEL)):
            rows.append({"method": method, "error_on": part, "mse_db": round(mse_db_on(est, x, idx), 2),
                         "q2": q2 if method.startswith("KF") else "", "r2": r2 if method.startswith("KF") else "",
                         "paper_kf_db": PAPER_DB["KF"], "n_train_windows": meta["n_train"],
                         "steps_1hz": meta["steps_1hz"]})
    return rows


def main() -> None:
    path = generate()
    _, meta = load_dataset(path)
    print(f"saved {path}")
    print(f"1 Hz steps: {meta['steps_1hz']} ({meta['duration_s'] / 60:.1f} min; paper: 5850, 5556 after cleaning)")
    print(f"windows: train {meta['n_train']} x {T_TRAIN}, val {N_VAL} x {T_VAL}, test 1 x {T_TEST} "
          f"(paper: 23 / 2 / 1)")
    rows = check()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nKF tuned on validation: q2 = {rows[0]['q2']:.3g}, r2 = {rows[0]['r2']:.3g}")
    print(f"{'method':30s} {'error on':11s} {'MSE [dB]':>9s}")
    for r in rows:
        print(f"{r['method']:30s} {r['error_on']:11s} {r['mse_db']:9.2f}")
    print(f"paper (Table X): KF {PAPER_DB['KF']} dB, KalmanNet {PAPER_DB['KalmanNet']} dB")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()