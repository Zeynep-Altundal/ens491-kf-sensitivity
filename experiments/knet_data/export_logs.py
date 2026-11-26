"""Export per-time-step filter logs for the KalmanNet benchmarks (input of the peak analysis).

What this script does
---------------------
For every benchmark dataset (or the ones matching ``--data``) it runs, on the test split and with
the model given to the filters, the

    - KF   (linear models only; computed with the EKF code, which equals the KF for linear models)
    - EKF
    - ESKF (vector-state error-state filter; equals the EKF here, kept so every data set has the
            same filter list as the drone experiments)

and, if a trained run exists (``kalmannet_run.py`` / ``sweep.py``), KalmanNet. Each filter gets one
log file, the generic counterpart of the run log in the common interface (section 5):

    ~/data/knet_logs/<dataset>_<given>/<FILTER>.npz
        (all arrays float32 except t)
        t        [T]          time of each step (s; step index x dt)
        x_hat    [N, T, m]    posterior estimate
        err      [N, T, m]    truth - estimate (error convention of the common interface)
        P        [N, T, m, m] posterior covariance (NaN for KalmanNet)
        nees     [N, T]       err^T P^-1 err (NaN for KalmanNet)
        innov    [N, T, n]    y_t - h(f(x_hat_{t-1})), computed the same way for every filter
        S        [N, T, n, n] innovation covariance (NaN for KalmanNet)
        nis      [N, T]       innov^T S^-1 innov (NaN for KalmanNet)
        meta     JSON         filter, dataset, recording file, source, given model, Q/R used,
                              state and measurement names, seed (KalmanNet)

The truth and the measurements stay in the recording (the dataset ``.npz``); ``meta.recording``
points to it. Noise covariances: the true Q and R for the synthetic data, and the q^2, r^2 chosen
on the validation windows for NCLT (``results/nclt_check.csv``).

Usage (from the repository root)::

    python -m experiments.knet_data.export_logs                       # all datasets
    python -m experiments.knet_data.export_logs --data "lorenz_rotH1_*" --seed 0
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path

import numpy as np

from experiments.filter_family import lorenz as zl

from . import nclt, sinusoidal
from .common import DATA_DIR, batched_ekf, batched_eskf, load_dataset
from .eskf_equivalence import linear_model, nclt_noise, sinusoidal_model
from .kalmannet_run import normalize_given, run_dir
from .lorenz import jacobian as lorenz_jacobian

LOG_DIR = Path(os.environ.get("KNET_LOG_DIR", DATA_DIR.parent / "knet_logs"))
SOURCE = {"linear": "kalmannet_linear", "sinusoidal": "kalmannet_sine", "lorenz": "kalmannet_lorenz",
          "nclt": "nclt"}
STATE_NAMES = {"linear": ["x1", "x2"], "sinusoidal": ["x1", "x2"], "lorenz": ["x", "y", "z"],
               "nclt": ["p_north", "v_north", "p_east", "v_east"]}
MEAS_NAMES = {"linear": ["y1", "y2"], "sinusoidal": ["y1", "y2"], "lorenz": ["y1", "y2", "y3"],
              "nclt": ["v_north", "v_east"]}


def filter_setup(arrays: dict, meta: dict, given: str) -> dict:
    """Model functions, noise covariances and initial state given to the filters."""
    bench = meta["benchmark"]
    if bench == "linear":
        return {"funcs": linear_model(arrays["F_model"], arrays["H_model"]), "Q": arrays["Q"],
                "R": arrays["R"], "x0": arrays["x0"], "P0": np.zeros_like(arrays["Q"]), "linear": True,
                "noise": "true Q, R"}
    if bench == "nclt":
        q2, r2 = nclt_noise()
        return {"funcs": linear_model(nclt.F, nclt.H), "Q": q2 * nclt.Q_STRUCT, "R": r2 * np.eye(2),
                "x0": arrays["x0_test"], "P0": np.zeros((4, 4)), "linear": True,
                "noise": f"q2={q2:g}, r2={r2:g} (validation grid search)"}
    if bench == "sinusoidal":
        return {"funcs": sinusoidal_model(sinusoidal.PARAMS[given]), "Q": arrays["Q"], "R": arrays["R"],
                "x0": arrays["x0"], "P0": np.zeros((2, 2)), "linear": False, "noise": "true Q, R"}
    J = {"J5": zl.J_DATA, "J2": zl.J_MISMATCH}[given]
    f, H = zl.make_f(J), arrays["H_model"]
    funcs = (f, lorenz_jacobian(f), lambda x: x @ H.T, lambda x: np.broadcast_to(H, (x.shape[0],) + H.shape))
    return {"funcs": funcs, "Q": arrays["Q"], "R": arrays["R"], "x0": arrays["x0"],
            "P0": zl.P0_SCALE * np.eye(3), "linear": False, "noise": "true Q, R"}


def quad_form(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """v^T M^-1 v over the leading axes; v (..., k), M (..., k, k)."""
    return np.einsum("...i,...i->...", v, np.linalg.solve(M, v[..., None])[..., 0])


def innovations(x_hat: np.ndarray, y: np.ndarray, x0: np.ndarray, f, h) -> np.ndarray:
    """y_t - h(f(x_hat_{t-1})) with x_hat_0 = x0, for estimates that come without innovations."""
    N, T, m = x_hat.shape
    prev = np.concatenate([np.broadcast_to(x0, (N, m))[:, None], x_hat[:, :-1]], axis=1)
    prior = f(prev.reshape(-1, m)).reshape(N, T, m)
    return y - h(prior.reshape(-1, m)).reshape(y.shape)


def write_log(path: Path, name: str, x_hat, P, innov, S, t, x_true, info: dict) -> Path:
    err = x_true - x_hat
    with np.errstate(all="ignore"):
        nees = quad_form(err, P) if P is not None else np.full(err.shape[:2], np.nan)
        nis = quad_form(innov, S) if S is not None else np.full(innov.shape[:2], np.nan)
    m, n = x_hat.shape[2], innov.shape[2]
    if P is None:
        P = np.full(x_hat.shape + (m,), np.nan)
    if S is None:
        S = np.full(innov.shape + (n,), np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    f32 = np.float32   # single precision keeps the Lorenz logs small; enough for analysis and plots
    np.savez_compressed(path, t=t, x_hat=x_hat.astype(f32), err=err.astype(f32), P=P.astype(f32),
                        nees=nees.astype(f32), innov=innov.astype(f32), S=S.astype(f32), nis=nis.astype(f32),
                        meta=json.dumps({"filter": name, **info}))
    return path


def export(data: str, given: str = "", seed: int = 0) -> list[Path]:
    rec_path = DATA_DIR / f"{data}.npz"
    arrays, meta = load_dataset(rec_path)
    bench = meta["benchmark"]
    given = normalize_given(bench, given)
    setup = filter_setup(arrays, meta, given)
    y, x = arrays["y_test"], arrays["x_test"]
    dt = float(meta.get("dt", 1.0))
    t = dt * np.arange(1, y.shape[1] + 1)
    folder = LOG_DIR / f"{data}_{given}"
    info = {"dataset": data, "recording": str(rec_path), "source": SOURCE[bench], "given_model": given,
            "noise_given": setup["noise"], "state_names": STATE_NAMES[bench], "meas_names": MEAS_NAMES[bench],
            "error_convention": "truth - estimate", "split": "test"}
    if bench == "nclt":
        info["frame"] = "NCLT local frame: x north, y east (not the ENU frame of the drone recordings)"

    args = (y, *setup["funcs"], setup["Q"], setup["R"], setup["x0"], setup["P0"])
    written = []
    with np.errstate(all="ignore"):
        ekf = batched_ekf(*args, return_innov=True)
        eskf = batched_eskf(*args, return_innov=True)
    filters = [("EKF", ekf), ("ESKF", eskf)]
    if setup["linear"]:
        filters.insert(0, ("KF", ekf))   # identical to the EKF for a linear model
    for name, (xh, P, nu, S) in filters:
        written.append(write_log(folder / f"{name}.npz", name, xh, P, nu, S, t, x, info))

    res = run_dir(data, given, seed) / "results.npz"
    if not res.exists() and seed == 0:   # runs made before the seed was added to the folder name
        res = run_dir(data, given, seed).with_name(f"kalmannet_{data}_{given}") / "results.npz"
    if res.exists():
        with np.load(res) as fh:
            xh = fh["x_hat"]
            run_meta = json.loads(str(fh["meta"]))
        f, _, h, _ = setup["funcs"]
        nu = innovations(xh, y, setup["x0"], f, h)
        kinfo = {**info, "seed": seed, "training": {k: run_meta.get(k) for k in ("steps", "lr", "wd", "best_val_db")}}
        written.append(write_log(folder / "KalmanNet.npz", "KalmanNet", xh, None, nu, None, t, x, kinfo))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="*", help="dataset name pattern (default: all)")
    ap.add_argument("--given", default="", help="sinusoidal: partial|full; lorenz: J5|J2 (default: partial / J5)")
    ap.add_argument("--seed", type=int, default=0, help="which KalmanNet run to export")
    a = ap.parse_args()
    names = sorted(p.stem for p in DATA_DIR.glob("*.npz") if fnmatch.fnmatch(p.stem, a.data))
    for data in names:
        paths = export(data, a.given, a.seed)
        print(f"{data}: " + ", ".join(p.stem for p in paths))
    print(f"\nlogs in {LOG_DIR}")


if __name__ == "__main__":
    main()
