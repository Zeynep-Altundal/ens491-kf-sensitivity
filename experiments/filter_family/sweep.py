"""One-parameter-at-a-time sensitivity study of the filter family.

Implements the protocol in ``experiments/filter_family/README.md``
("Sensitivity study protocol"). Writes ``results/sweep_<name>.csv``.

Usage (from the repository root)::

    python -m experiments.filter_family.sweep
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .filters import EKF, IEKF, UKF, AdaptiveEKF, run_filter
from .models import StateSpaceModel, range_bearing_model, simulate

N_RUNS = 100
STEPS = 60
SEED = 0
ADAPT_WINDOW = 20
DIVERGENCE_FACTOR = 10.0
FILTER_NAMES = ("EKF", "IEKF", "UKF", "AdaptiveEKF")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

S_GRID = np.logspace(-2, 2, 9)
BEARING_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
SIGMA_P0_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0)
CLOSEST_APPROACH_GRID = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)


@dataclass(frozen=True)
class Setting:
    """True model plus the filter values of Q, R and P0 for one grid point."""

    true_model: StateSpaceModel
    Q: np.ndarray
    R: np.ndarray
    P0: np.ndarray


@dataclass(frozen=True)
class Sweep:
    name: str
    parameter: str
    grid: tuple
    make_setting: Callable[[float], Setting]
    log_x: bool = True


def _mistuning(which: str) -> Callable[[float], Setting]:
    def make(s: float) -> Setting:
        m = range_bearing_model()
        Q, R, P0 = m.Q, m.R, m.P0
        if which == "Q":
            Q = s * Q
        elif which == "R":
            R = s * R
        else:
            P0 = s * P0
        return Setting(m, Q, R, P0)
    return make


def _true(m: StateSpaceModel) -> Setting:
    return Setting(m, m.Q, m.R, m.P0)


SWEEPS = (
    Sweep("mistune_q", "Q scale s", tuple(S_GRID), _mistuning("Q")),
    Sweep("mistune_r", "R scale s", tuple(S_GRID), _mistuning("R")),
    Sweep("mistune_p0", "P0 scale s", tuple(S_GRID), _mistuning("P0")),
    Sweep("bearing_noise", "bearing noise std [rad]", BEARING_GRID,
          lambda v: _true(range_bearing_model(sigma_bearing=v))),
    Sweep("initial_uncertainty", "initial position std [m]", SIGMA_P0_GRID,
          lambda v: _true(range_bearing_model(p0_diag=(v**2, v**2, 1.0, 1.0)))),
    Sweep("closest_approach", "closest-approach distance d [m]", CLOSEST_APPROACH_GRID,
          lambda v: _true(range_bearing_model(q=1e-4, x0=(-30.0, v, 1.0, 0.0),
                                              p0_diag=(1.0, 1.0, 0.0025, 0.0025)))),
)


def _make_filter(name: str, model: StateSpaceModel, st: Setting):
    kwargs = dict(Q=st.Q, R=st.R)
    if name == "EKF":
        return EKF(model, model.x0, st.P0, **kwargs)
    if name == "IEKF":
        return IEKF(model, model.x0, st.P0, **kwargs)
    if name == "UKF":
        return UKF(model, model.x0, st.P0, **kwargs)
    if name == "AdaptiveEKF":
        return AdaptiveEKF(model, model.x0, st.P0, window=ADAPT_WINDOW, **kwargs)
    raise ValueError(name)


def evaluate_setting(st: Setting) -> tuple[dict, dict]:
    """Run all filters on the same simulated data; returns per-filter metrics and geometry info."""
    m = st.true_model
    x_true, y = simulate(m, STEPS, N_RUNS, np.random.default_rng(SEED))
    sig_r, sig_b = np.sqrt(m.R[0, 0]), np.sqrt(m.R[1, 1])
    ranges = np.hypot(x_true[:, 1:, 0], x_true[:, 1:, 1])
    e_fix = np.sqrt(sig_r**2 + (ranges[:, -1] * sig_b) ** 2)
    geometry = {"median_min_range": float(np.median(ranges.min(axis=1)))}

    metrics = {}
    for name in FILTER_NAMES:
        sq_err, nees, nis, iters = [], [], [], []
        diverged = 0
        for i in range(N_RUNS):
            filt = _make_filter(name, m, st)
            try:
                with np.errstate(all="ignore"):
                    out = run_filter(filt, y[i])
                    e = x_true[i, 1:] - out["x"]
                    finite = np.all(np.isfinite(out["x"])) and np.all(np.isfinite(out["P"]))
                    if not finite or np.linalg.norm(e[-1, :2]) > DIVERGENCE_FACTOR * e_fix[i]:
                        raise FloatingPointError("diverged")
                    run_nees = np.einsum("ti,tij,tj->t", e, np.linalg.inv(out["P"]), e)
                    run_nis = np.einsum("ti,tij,tj->t", out["nu"], np.linalg.inv(out["S"]), out["nu"])
            except (np.linalg.LinAlgError, FloatingPointError):
                diverged += 1
                continue
            sq_err.append(np.sum(e[:, :2] ** 2, axis=1))
            nees.append(run_nees)
            nis.append(run_nis)
            if isinstance(filt, IEKF):
                iters.extend(filt.iterations)
        ok = N_RUNS - diverged
        metrics[name] = {
            "pos_rmse": float(np.sqrt(np.mean(sq_err))) if ok else float("nan"),
            "mean_nees": float(np.mean(nees)) if ok else float("nan"),
            "mean_nis": float(np.mean(nis)) if ok else float("nan"),
            "divergence_rate": diverged / N_RUNS,
            "mean_iekf_iterations": float(np.mean(iters)) if iters else float("nan"),
        }
    return metrics, geometry


def run_sweep(sweep: Sweep) -> list[dict]:
    rows = []
    for value in sweep.grid:
        metrics, geometry = evaluate_setting(sweep.make_setting(value))
        for name in FILTER_NAMES:
            rows.append({"value": float(value), "filter": name, **metrics[name], **geometry})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    for sweep in SWEEPS:
        rows = run_sweep(sweep)
        path = RESULTS_DIR / f"sweep_{sweep.name}.csv"
        write_csv(rows, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
