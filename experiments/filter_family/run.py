"""Nominal Monte Carlo comparison on the range-bearing benchmark.

All filters are given the true Q and R and start from ``(x0_true, P0_true)``
while the true initial state is drawn from ``N(x0_true, P0_true)``. The
adaptive filter starts with the true R and adapts it (window 20).

Outputs in ``results/``:

* ``nominal_comparison.csv``: per filter position RMSE, mean NEES, mean NIS,
  runtime per step, number of runs with a final position error above 10 m,
  mean IEKF iterations;
* ``nominal_error_over_time.csv``: position RMSE over the runs at each step;
* ``nominal_example_run.csv``: truth, measurements and all estimates of the
  first run whose true bearing crosses +pi / -pi (the example run).

Usage (from the repository root)::

    python -m experiments.filter_family.run
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .filters import EKF, IEKF, UKF, AdaptiveEKF, run_filter
from .models import range_bearing_model, simulate

N_RUNS = 100
STEPS = 60
SEED = 0
ADAPT_WINDOW = 20
DIVERGENCE_M = 10.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def make_filters(model):
    return {
        "EKF": lambda: EKF(model, model.x0, model.P0),
        "IEKF": lambda: IEKF(model, model.x0, model.P0),
        "UKF": lambda: UKF(model, model.x0, model.P0),
        "AdaptiveEKF": lambda: AdaptiveEKF(model, model.x0, model.P0, window=ADAPT_WINDOW),
    }


def run_comparison(n_runs: int = N_RUNS, steps: int = STEPS, seed: int = SEED):
    model = range_bearing_model()
    x_true, y = simulate(model, steps, n_runs, np.random.default_rng(seed))
    makers = make_filters(model)
    true_bearing = np.arctan2(x_true[:, :, 1] - 0.0, x_true[:, :, 0] - 0.0)
    wraps = np.any(np.abs(np.diff(true_bearing, axis=1)) > np.pi, axis=1)
    example_idx = int(np.argmax(wraps)) if wraps.any() else 0
    summary, err_t, example = {}, {}, {"index": example_idx, "n_wrap_runs": int(wraps.sum())}
    for name, make in makers.items():
        pos_err = np.empty((n_runs, steps))
        nees = np.empty((n_runs, steps))
        nis = np.empty((n_runs, steps))
        seconds, iters = 0.0, []
        for i in range(n_runs):
            filt = make()
            out = run_filter(filt, y[i])
            e = x_true[i, 1:] - out["x"]
            pos_err[i] = np.linalg.norm(e[:, :2], axis=1)
            nees[i] = np.einsum("ti,tij,tj->t", e, np.linalg.inv(out["P"]), e)
            nis[i] = np.einsum("ti,tij,tj->t", out["nu"], np.linalg.inv(out["S"]), out["nu"])
            seconds += out["seconds"]
            if isinstance(filt, IEKF):
                iters.extend(filt.iterations)
            if i == example_idx:
                example[name] = out["x"]
        summary[name] = {
            "pos_rmse": float(np.sqrt(np.mean(pos_err**2))),
            "mean_nees": float(np.mean(nees)),
            "mean_nis": float(np.mean(nis)),
            "runtime_us_per_step": 1e6 * seconds / (n_runs * steps),
            "runs_final_pos_err_gt_10m": int(np.sum(pos_err[:, -1] > DIVERGENCE_M)),
            "mean_iterations": float(np.mean(iters)) if iters else float("nan"),
        }
        err_t[name] = np.sqrt(np.mean(pos_err**2, axis=0))
    return model, x_true, y, summary, err_t, example


def write_outputs(model, x_true, y, summary, err_t, example, out_dir: Path = RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "nominal_comparison.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        keys = list(next(iter(summary.values())))
        writer.writerow(["filter"] + keys)
        for name, row in summary.items():
            writer.writerow([name] + [f"{row[k]:.10g}" for k in keys])

    steps = len(next(iter(err_t.values())))
    t = model.dt * np.arange(1, steps + 1)
    names = list(err_t)
    np.savetxt(out_dir / "nominal_error_over_time.csv",
               np.column_stack([t] + [err_t[n] for n in names]), delimiter=",",
               header=",".join(["t"] + [f"pos_rmse_{n}" for n in names]), comments="", fmt="%.10g")

    # Measurements converted to Cartesian coordinates for plotting only.
    i = example["index"]
    meas_xy = np.column_stack([y[i, :, 0] * np.cos(y[i, :, 1]), y[i, :, 0] * np.sin(y[i, :, 1])])
    cols = [t, np.full(steps, i), x_true[i, 1:, 0], x_true[i, 1:, 1], meas_xy[:, 0], meas_xy[:, 1]]
    header = ["t", "run", "px_true", "py_true", "meas_x", "meas_y"]
    for n in names:
        cols += [example[n][:, 0], example[n][:, 1]]
        header += [f"px_{n}", f"py_{n}"]
    np.savetxt(out_dir / "nominal_example_run.csv", np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.10g")


def main() -> None:
    model, x_true, y, summary, err_t, example = run_comparison()
    write_outputs(model, x_true, y, summary, err_t, example)
    lo_x, hi_x = chi2.ppf([0.025, 0.975], df=model.n * N_RUNS) / N_RUNS
    lo_z, hi_z = chi2.ppf([0.025, 0.975], df=model.m * N_RUNS) / N_RUNS
    print(f"N = {N_RUNS} runs, {STEPS} steps; expected NEES {model.n} "
          f"(95 % interval of an {N_RUNS}-run mean [{lo_x:.3f}, {hi_x:.3f}]), expected NIS {model.m} "
          f"([{lo_z:.3f}, {hi_z:.3f}])")
    print(f"runs whose true bearing crosses +pi / -pi: {example['n_wrap_runs']} of {N_RUNS}; "
          f"example run = {example['index']}")
    print(f"{'filter':12}{'pos RMSE [m]':>14}{'mean NEES':>11}{'mean NIS':>10}{'us/step':>10}"
          f"{'final err>10m':>15}{'iterations':>12}")
    for name, r in summary.items():
        print(f"{name:12}{r['pos_rmse']:14.4f}{r['mean_nees']:11.4f}{r['mean_nis']:10.4f}"
              f"{r['runtime_us_per_step']:10.1f}{r['runs_final_pos_err_gt_10m']:15d}"
              f"{r['mean_iterations']:12.3f}")
    for name in ("nominal_comparison.csv", "nominal_error_over_time.csv", "nominal_example_run.csv"):
        print(f"wrote {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
