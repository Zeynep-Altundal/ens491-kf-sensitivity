"""Lorenz-attractor benchmark of the KalmanNet paper, evaluated with the classical filters.

Implements the protocol "Lorenz-attractor benchmark of the KalmanNet paper" in
``experiments/filter_family/README.md``. The model is implemented here from the equations of the
paper (arXiv:2107.10043v3, eqs. (18)-(21)) and the parameter values documented in the README; no
source file of the reference code is used. KalmanNet itself is not trained or run.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.filter_family.lorenz
"""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat

from .filters import EKF, IEKF, UKF, AdaptiveEKF, run_filter
from .models import StateSpaceModel

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

DT = 0.02
J_DATA = 5
J_MISMATCH = 2
NU_DB = -20.0
N_RUNS = 100
STEPS = 2000
SEED = 0
P0_SCALE = 1e-10
FD_STEP = 1e-6
ADAPT_WINDOW = 20
X0 = np.array([1.0, 1.0, 1.0])
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

# (name, J used by the filters, list of 1/r^2 [dB]); data always use J_DATA.
SETTINGS = [("full", J_DATA, 1.0 / 10 ** (d / 10.0), d) for d in (0.0, 10.0, 20.0, 30.0, 40.0)] + \
           [("mismatch_J2", J_MISMATCH, 1.0 / 10 ** (d / 10.0), d) for d in (10.0, 20.0, 30.0, 40.0)]
FILTERS = ("EKF", "EKF-F", "IEKF", "UKF", "AdaptiveEKF")

# Values transcribed from arXiv:2107.10043v3: (table, setting, filter, 1/r^2 [dB], MSE [dB], spread [dB]).
PAPER = [("Table V", "full", f, d, v, None) for f, vals in
         (("EKF", (-10.45, -20.37, -30.40, -40.39, -49.89)),
          ("UKF", (-5.62, -12.04, -20.45, -30.05, -40.00)))
         for d, v in zip((0.0, 10.0, 20.0, 30.0, 40.0), vals)]
PAPER += [("Table VII", "full", "EKF", d, v, s) for d, v, s in
          zip((10.0, 20.0, 30.0, 40.0), (-20.37, -30.40, -40.39, -49.89), (0.25, 0.24, 0.24, 0.20))]
PAPER += [("Table VII", "mismatch_J2", "EKF", d, v, s) for d, v, s in
          zip((10.0, 20.0, 30.0, 40.0), (-19.47, -23.63, -33.51, -41.15), (0.25, 0.11, 0.18, 0.12))]
PAPER += [("Table VII", "mismatch_J2", "UKF", d, v, s) for d, v, s in
          zip((10.0, 20.0, 30.0, 40.0), (-11.95, -20.45, -30.05, -39.98), (0.87, 0.27, 0.09, 0.09))]


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------

def lorenz_A(x: np.ndarray, form: str = "code") -> np.ndarray:
    """State-dependent matrix A(x) with dx/dt = A(x) x for the Lorenz system, batched over (..., 3).

    ``form="code"``: the placement of the reference code, [[-s, s, 0], [r - x3, -1, 0], [x2, 0, -b]].
    ``form="paper"``: the placement of eq. (18) of the paper, [[-s, s, 0], [r, -1, -x1], [0, x1, -b]].
    Both give the same vector field A(x) x.
    """
    x = np.asarray(x, dtype=float)
    A = np.zeros(x.shape[:-1] + (3, 3))
    A[..., 0, 0], A[..., 0, 1] = -SIGMA, SIGMA
    A[..., 1, 1], A[..., 2, 2] = -1.0, -BETA
    if form == "code":
        A[..., 1, 0] = RHO - x[..., 2]
        A[..., 2, 0] = x[..., 1]
    elif form == "paper":
        A[..., 1, 0] = RHO
        A[..., 1, 2] = -x[..., 0]
        A[..., 2, 1] = x[..., 0]
    else:
        raise ValueError(form)
    return A


def taylor_F(x: np.ndarray, J: int, dt: float = DT, form: str = "code") -> np.ndarray:
    """F(x) = I + sum_{j=1..J} (A(x) dt)^j / j!, batched."""
    M = lorenz_A(x, form) * dt
    F = np.broadcast_to(np.eye(3), M.shape).copy()
    term = F.copy()
    for j in range(1, J + 1):
        term = term @ M / j
        F = F + term
    return F


def make_f(J: int):
    def f(x):
        x = np.asarray(x, dtype=float)
        return np.einsum("...ij,...j->...i", taylor_F(x, J), x)
    return f


def jacobian_exact(J: int):
    """Jacobian of f(x) = F(x) x by central differences."""
    f = make_f(J)

    def jac(x):
        x = np.asarray(x, dtype=float)
        cols = []
        for k in range(3):
            d = np.zeros(3)
            d[k] = FD_STEP
            cols.append((f(x + d) - f(x - d)) / (2.0 * FD_STEP))
        return np.stack(cols, axis=-1)
    return jac


def jacobian_F(J: int):
    """F(x) used as the Jacobian (convention of the reference code)."""
    return lambda x: taylor_F(np.asarray(x, dtype=float), J)


def lorenz_model(r2: float, J: int, jacobian: str = "exact") -> StateSpaceModel:
    q2 = r2 * 10 ** (NU_DB / 10.0)
    jac = jacobian_exact(J) if jacobian == "exact" else jacobian_F(J)
    return StateSpaceModel(
        name="lorenz", f=make_f(J), F_jacobian=jac,
        h=lambda x: np.asarray(x, dtype=float), H_jacobian=lambda x: np.eye(3),
        Q=q2 * np.eye(3), R=r2 * np.eye(3), x0=X0.copy(), P0=P0_SCALE * np.eye(3),
        residual=lambda a, b: np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
        measurement_mean=lambda Z, w: w @ Z, dt=DT)


def simulate(r2: float, n_traj: int, steps: int, rng: np.random.Generator, J: int = J_DATA):
    """x_{t+1} = F(x_t) x_t + w_t, y_t = x_t + v_t from the fixed x_0; returns x_1..x_T, y_1..y_T."""
    q2 = r2 * 10 ** (NU_DB / 10.0)
    f = make_f(J)
    w = rng.standard_normal((n_traj, steps, 3)) * math.sqrt(q2)
    v = rng.standard_normal((n_traj, steps, 3)) * math.sqrt(r2)
    x = np.empty((n_traj, steps, 3))
    prev = np.broadcast_to(X0, (n_traj, 3))
    for t in range(steps):
        prev = f(prev) + w[:, t]
        x[:, t] = prev
    return x, x + v


# --------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------

def make_filter(name: str, model: StateSpaceModel, model_F: StateSpaceModel):
    if name == "EKF":
        return EKF(model, model.x0, model.P0)
    if name == "EKF-F":
        return EKF(model_F, model_F.x0, model_F.P0)
    if name == "IEKF":
        return IEKF(model, model.x0, model.P0)
    if name == "UKF":
        return UKF(model, model.x0, model.P0)
    if name == "AdaptiveEKF":
        return AdaptiveEKF(model, model.x0, model.P0, window=ADAPT_WINDOW)
    raise ValueError(name)


def run_one(args):
    k, fname, i, x, y = args
    _, J, r2, _ = SETTINGS[k]
    model, model_F = lorenz_model(r2, J, "exact"), lorenz_model(r2, J, "F")
    try:
        with np.errstate(all="ignore"):
            o = run_filter(make_filter(fname, model, model_F), y)
            if not (np.all(np.isfinite(o["x"])) and np.all(np.isfinite(o["P"]))):
                raise FloatingPointError
            e = x - o["x"]
            nees = np.einsum("ti,tij,tj->t", e, np.linalg.inv(o["P"]), e)
            nis = np.einsum("ti,tij,tj->t", o["nu"], np.linalg.inv(o["S"]), o["nu"])
            res = {"mse": float(np.mean(e**2)), "sq": float(np.mean(np.sum(e**2, axis=1))),
                   "nees": float(np.mean(nees)), "nis": float(np.mean(nis)), "ok": True,
                   "x_est": o["x"][:400] if i == 0 else None}
    except (np.linalg.LinAlgError, FloatingPointError):
        res = {"ok": False, "x_est": None}
    return k, fname, i, res


def db(m) -> float:
    return float(10.0 * np.log10(np.mean(m)))


def write_rows(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    data = []
    for k, (_, _, r2, _) in enumerate(SETTINGS):
        data.append(simulate(r2, N_RUNS, STEPS, np.random.default_rng([SEED, k])))
    tasks = [(k, f, i, data[k][0][i], data[k][1][i]) for k in range(len(SETTINGS)) for f in FILTERS
             for i in range(N_RUNS)]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tasks)} filter runs on {workers} processes", flush=True)
    raw: dict = {}
    with mp.Pool(workers) as pool:
        for k, f, i, res in pool.imap_unordered(run_one, tasks, chunksize=4):
            raw.setdefault((k, f), {})[i] = res

    summary = []
    for k, (sname, J, r2, d) in enumerate(SETTINGS):
        for f in FILTERS:
            runs = [raw[(k, f)][i] for i in range(N_RUNS)]
            ok = [r for r in runs if r["ok"]]
            row = {"setting": sname, "filter_J": J, "inv_r2_dB": d, "nu_dB": NU_DB, "filter": f,
                   "n_failed": N_RUNS - len(ok)}
            if ok:
                mse = np.array([r["mse"] for r in ok])
                est, lo, hi = bootstrap(db, mse)
                row.update({"mse_dB": est, "mse_dB_lo": lo, "mse_dB_hi": hi,
                            "mse_dB_paper_spread": float(10 * np.log10(np.std(mse, ddof=1) + np.mean(mse)) - est)})
                for key, name, stat in (("sq", "rmse", lambda a: float(np.sqrt(np.mean(a)))),
                                        ("nees", "mean_nees", mean_stat), ("nis", "mean_nis", mean_stat)):
                    est, lo, hi = bootstrap(stat, np.array([r[key] for r in ok]))
                    row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            summary.append(row)
    RESULTS_DIR.mkdir(exist_ok=True)
    write_rows(summary, RESULTS_DIR / "lorenz_summary.csv")
    write_rows([{"source": t, "setting": s, "filter": f, "inv_r2_dB": d, "mse_dB": v,
                 "spread_dB": ("" if sp is None else sp)} for t, s, f, d, v, sp in PAPER],
               RESULTS_DIR / "lorenz_paper_values.csv")
    comp = []
    for t, s, f, d, v, sp in PAPER:
        ours = next(r for r in summary if r["setting"] == s and r["filter"] == f and r["inv_r2_dB"] == d)
        row = {"paper_table": t, "setting": s, "filter": f, "inv_r2_dB": d, "paper_mse_dB": v,
               "paper_spread_dB": ("" if sp is None else sp), "n_failed": ours["n_failed"]}
        if "mse_dB" in ours:
            row.update({"ours_mse_dB": ours["mse_dB"], "ours_lo": ours["mse_dB_lo"], "ours_hi": ours["mse_dB_hi"],
                        "ours_minus_paper_dB": ours["mse_dB"] - v,
                        "paper_inside_our_interval": ours["mse_dB_lo"] <= v <= ours["mse_dB_hi"]})
        comp.append(row)
    write_rows(comp, RESULTS_DIR / "lorenz_comparison.csv")

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, sname, title in ((axes[0], "full", "full information (J = 5)"),
                             (axes[1], "mismatch_J2", "state-evolution mismatch (filters J = 2)")):
        for f in FILTERS:
            rr = [r for r in summary if r["setting"] == sname and r["filter"] == f and "mse_dB" in r]
            ax.plot([r["inv_r2_dB"] for r in rr], [r["mse_dB"] for r in rr], "o-", label=f"{f} (this study)")
        for f, mk in (("EKF", "k^"), ("UKF", "kv")):
            pp = [p for p in PAPER if p[1] == sname and p[2] == f and p[0] == ("Table V" if sname == "full" else "Table VII")]
            ax.plot([p[3] for p in pp], [p[4] for p in pp], mk, ms=8, mfc="none", label=f"{f} (paper)")
        ax.set_xlabel("1/r^2 [dB]")
        ax.set_ylabel("MSE [dB]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "lorenz_mse.png", dpi=120)
    plt.close(fig)

    k = next(i for i, s in enumerate(SETTINGS) if s[0] == "full" and s[3] == 20.0)
    x, y = data[k][0][0], data[k][1][0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    t = np.arange(1, 401) * DT
    for j, ax in enumerate(axes):
        ax.plot(t, y[:400, j], ".", ms=2, color="0.7", label="observation")
        ax.plot(t, x[:400, j], "k-", lw=1, label="truth")
        for f in ("EKF", "UKF"):
            est = raw[(k, f)][0]["x_est"]
            if est is not None:
                ax.plot(t, est[:, j], lw=0.8, label=f)
        ax.set_ylabel(f"x{j + 1}")
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title("Lorenz attractor, run 0, 1/r^2 = 20 dB, nu = -20 dB (first 400 steps)")
    axes[-1].set_xlabel("time [s]")
    fig.savefig(FIGURES_DIR / "lorenz_example.png", dpi=120)
    plt.close(fig)
    for r in summary:
        print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()
               if k2 in ("setting", "inv_r2_dB", "filter", "n_failed", "mse_dB", "mse_dB_lo", "mse_dB_hi",
                         "mean_nees", "mean_nis")})


if __name__ == "__main__":
    main()
