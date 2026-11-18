"""Exploratory: classical filters next to a KalmanNet reference run on the Lorenz benchmark.

Implements the protocol "Exploratory KalmanNet baseline on the Lorenz benchmark" in
``experiments/filter_family/README.md``. KalmanNet and the reference EKF are run from the reference
repository outside this repository (see the README); this script

* reads the test trajectories exported by that run as plain ``.npy`` arrays (``allow_pickle=False``),
* runs the filters of this repository on them and on an independent realisation generated with the
  same parameters,
* parses the training log of the reference run (validation curve, reference test results).

Usage (from the repository root)::

    python -m experiments.filter_family.lorenz_knet_baseline <folder with the exported .npy files and the log>
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat

from .filters import EKF, IEKF, UKF, run_filter
from .lorenz import DT, FIGURES_DIR, RESULTS_DIR, X0, jacobian_exact, jacobian_F, make_f, write_rows
from .models import StateSpaceModel

R2 = 0.1
NU_DB = -20.0
Q2 = R2 * 10 ** (NU_DB / 10.0)
J = 5
STEPS = 100
N_OWN = 200
SEED = 0
P0_SCALE = 1e-10
ROT_DEG = 1.0
FILTERS = ("EKF", "EKF-F", "IEKF", "UKF")


def rotation_1deg() -> np.ndarray:
    """Rz(a) Ry(a) Rx(a) with a = 1 deg (rotation of the identity observation about x, y and z)."""
    a = math.radians(ROT_DEG)
    c, s = math.cos(a), math.sin(a)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


H_ROT = rotation_1deg()


def model(jacobian: str) -> StateSpaceModel:
    jac = jacobian_exact(J) if jacobian == "exact" else jacobian_F(J)
    return StateSpaceModel(
        name="lorenz_rot", f=make_f(J), F_jacobian=jac,
        h=lambda x: np.asarray(x, dtype=float) @ H_ROT.T, H_jacobian=lambda x: H_ROT,
        Q=Q2 * np.eye(3), R=R2 * np.eye(3), x0=X0.copy(), P0=P0_SCALE * np.eye(3),
        residual=lambda a, b: np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
        measurement_mean=lambda Z, w: w @ Z, dt=DT)


def simulate(n: int, rng: np.random.Generator):
    f = make_f(J)
    w = rng.standard_normal((n, STEPS, 3)) * math.sqrt(Q2)
    v = rng.standard_normal((n, STEPS, 3)) * math.sqrt(R2)
    x = np.empty((n, STEPS, 3))
    prev = np.broadcast_to(X0, (n, 3))
    for t in range(STEPS):
        prev = f(prev) + w[:, t]
        x[:, t] = prev
    return x, x @ H_ROT.T + v


def evaluate(x: np.ndarray, y: np.ndarray, dataset: str) -> list[dict]:
    rows = []
    for name in FILTERS:
        m = model("F" if name == "EKF-F" else "exact")
        cls = {"EKF": EKF, "EKF-F": EKF, "IEKF": IEKF, "UKF": UKF}[name]
        mse, nees, nis, failed = [], [], [], 0
        for i in range(len(x)):
            try:
                with np.errstate(all="ignore"):
                    o = run_filter(cls(m, m.x0, m.P0), y[i])
                if not np.all(np.isfinite(o["x"])):
                    raise FloatingPointError
            except (np.linalg.LinAlgError, FloatingPointError):
                failed += 1
                continue
            e = x[i] - o["x"]
            mse.append(np.mean(e**2))
            nees.append(np.mean(np.einsum("ti,tij,tj->t", e, np.linalg.inv(o["P"]), e)))
            nis.append(np.mean(np.einsum("ti,tij,tj->t", o["nu"], np.linalg.inv(o["S"]), o["nu"])))
        mse = np.array(mse)
        db = lambda a: float(10 * np.log10(np.mean(a)))
        est, lo, hi = bootstrap(db, mse)
        row = {"source": "this repository", "dataset": dataset, "method": name, "n_trajectories": len(mse),
               "n_failed": failed, "mse_dB": est, "mse_dB_lo": lo, "mse_dB_hi": hi,
               "spread_dB": float(10 * np.log10(np.std(mse, ddof=1) + np.mean(mse)) - est)}
        for key, vals in (("mean_nees", nees), ("mean_nis", nis)):
            e2, l2, h2 = bootstrap(mean_stat, np.array(vals))
            row.update({key: e2, f"{key}_lo": l2, f"{key}_hi": h2})
        rows.append(row)
    return rows


def parse_log(path: Path) -> dict:
    text = path.read_text()
    num = r"tensor\((-?[0-9.eE+-]+|nan|inf)\)"
    curve = [(int(a), float(b), float(c)) for a, b, c in
             re.findall(rf"^(\d+) MSE Training : {num} \[dB\] MSE Validation : {num} \[dB\]", text, re.M)]

    def grab(label):
        m = re.search(rf"{re.escape(label)} {num} \[dB\]", text)
        return float(m.group(1)) if m else float("nan")
    opt = re.findall(r"Optimal idx: (\d+) Optimal : " + num, text)
    return {"curve": curve,
            "ekf_mse": grab("Extended Kalman Filter - MSE LOSS:"), "ekf_std": grab("Extended Kalman Filter - STD:"),
            "knet_mse": grab("KalmanNet - MSE LOSS:") if "KalmanNet - MSE LOSS:" in text else grab("KNet - MSE LOSS:"),
            "knet_std": grab("KalmanNet - STD:") if "KalmanNet - STD:" in text else grab("KNet - STD:"),
            "obs_mse": grab("Observation Noise Floor(test dataset) - MSE LOSS:"),
            "obs_std": grab("Observation Noise Floor(test dataset) - STD:"),
            "opt": (int(opt[-1][0]), float(opt[-1][1])) if opt else None,
            "test_lines": [l for l in text.splitlines() if "MSE LOSS" in l or " - STD" in l or "Inference Time" in l]}


def main() -> None:
    folder = Path(sys.argv[1])
    xi = np.load(folder / "kalmannet_test_target.npy", allow_pickle=False)
    yi = np.load(folder / "kalmannet_test_input.npy", allow_pickle=False)
    x_same = np.transpose(xi, (0, 2, 1)).astype(float)  # (N, m, T) -> (N, T, m)
    y_same = np.transpose(yi, (0, 2, 1)).astype(float)
    rows = evaluate(x_same, y_same, "same test trajectories (exported)")
    x_own, y_own = simulate(N_OWN, np.random.default_rng(SEED))
    rows += evaluate(x_own, y_own, "independent realisation (seed 0)")
    log = parse_log(folder / "knet_train.log")
    ref = [{"source": "reference repository", "dataset": "reference test set", "method": "EKF (reference code)",
            "n_trajectories": len(x_same), "mse_dB": log["ekf_mse"], "spread_dB": log["ekf_std"]},
           {"source": "reference repository", "dataset": "reference test set", "method": "observation noise floor",
            "n_trajectories": len(x_same), "mse_dB": log["obs_mse"], "spread_dB": log["obs_std"]}]
    if not math.isnan(log["knet_mse"]):
        ref.append({"source": "reference repository", "dataset": "reference test set", "method": "KalmanNet",
                    "n_trajectories": len(x_same), "mse_dB": log["knet_mse"], "spread_dB": log["knet_std"]})
    write_rows(ref + rows, RESULTS_DIR / "lorenz_knet_baseline.csv")
    curve = log["curve"]
    write_rows([{"step": s, "train_mse_dB": a, "validation_mse_dB": b} for s, a, b in curve],
               RESULTS_DIR / "lorenz_knet_training_curve.csv")
    if curve:
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        ax.plot([c[0] for c in curve], [c[1] for c in curve], lw=0.6, color="0.6", label="training batch")
        ax.plot([c[0] for c in curve], [c[2] for c in curve], lw=1.0, color="C0", label="validation (100 trajectories)")
        ax.axhline(log["ekf_mse"], color="C3", ls="--", lw=1, label="reference EKF on the test set")
        ax.set_xlabel("training step")
        ax.set_ylabel("MSE [dB]")
        ax.set_title("Exploratory KalmanNet run on the Lorenz benchmark (reference code, default settings)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(FIGURES_DIR / "lorenz_knet_training_curve.png", dpi=120)
        plt.close(fig)
    for r in ref + rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("dataset", "method", "n_failed", "mse_dB", "mse_dB_lo", "mse_dB_hi", "spread_dB",
                        "mean_nees", "mean_nis")})
    print("steps logged:", len(curve), "optimum:", log["opt"])
    print("\n".join(log["test_lines"]))


if __name__ == "__main__":
    main()
