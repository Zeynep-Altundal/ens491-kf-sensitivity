"""Figures of the nominal filter-family comparison.

Reads the CSVs written by :mod:`experiments.filter_family.run` and
:mod:`experiments.filter_family.sweep` and writes PNGs to
``experiments/filter_family/figures/``. Usage (from the repository root)::

    python -m experiments.filter_family.run
    python -m experiments.filter_family.sweep
    python -m experiments.filter_family.plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .run import RESULTS_DIR  # noqa: E402
from .sweep import SWEEPS  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
FILTERS = ("EKF", "IEKF", "UKF", "AdaptiveEKF")
STYLES = {"EKF": ("C0", "-"), "IEKF": ("C1", "--"), "UKF": ("C2", "-."), "AdaptiveEKF": ("C3", ":")}


def _load(name: str) -> dict[str, np.ndarray]:
    table = np.genfromtxt(RESULTS_DIR / name, delimiter=",", names=True)
    return {k: table[k] for k in table.dtype.names}


def plot_example(path: Path) -> None:
    d = _load("nominal_example_run.csv")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(d["meas_x"], d["meas_y"], ".", color="0.6", ms=4, label="measurement (range, bearing)")
    ax.plot(d["px_true"], d["py_true"], "k-", lw=2, label="truth")
    for n in FILTERS:
        c, ls = STYLES[n]
        ax.plot(d[f"px_{n}"], d[f"py_{n}"], color=c, ls=ls, lw=1.3, label=n)
    ax.plot(0, 0, "k^", ms=9, label="sensor")
    # Mark where the true bearing wraps (crossing of the negative x axis of the sensor).
    b = np.arctan2(d["py_true"], d["px_true"])
    jumps = np.nonzero(np.abs(np.diff(b)) > np.pi)[0]
    for j, k in enumerate(jumps):
        xm = 0.5 * (d["px_true"][k] + d["px_true"][k + 1])
        ym = 0.5 * (d["py_true"][k] + d["py_true"][k + 1])
        ax.plot(xm, ym, "mo", ms=10, mfc="none", mew=1.5,
                label="true bearing wraps +pi / -pi" if j == 0 else None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"range-bearing benchmark: example run (run {int(d['run'][0])})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_over_time(path: Path) -> None:
    d = _load("nominal_error_over_time.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n in FILTERS:
        c, ls = STYLES[n]
        ax.plot(d["t"], d[f"pos_rmse_{n}"], color=c, ls=ls, lw=1.5, label=n)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position RMSE over runs [m]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("range-bearing benchmark: position error over time (Monte Carlo)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Reference value of the swept parameter in the nominal benchmark (vertical line), if any.
SWEEP_REFERENCE = {"mistune_q": 1.0, "mistune_r": 1.0, "mistune_p0": 1.0,
                   "bearing_noise": 0.05, "initial_uncertainty": 5.0}
EXPECTED_NEES = 4.0


def _load_sweep(name: str) -> dict[str, dict[str, np.ndarray]]:
    table = np.genfromtxt(RESULTS_DIR / f"sweep_{name}.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    out = {}
    for f in FILTERS:
        rows = table[table["filter"] == f]
        out[f] = {k: np.asarray(rows[k], dtype=float) for k in table.dtype.names if k != "filter"}
    return out


def plot_sweep(sweep, path: Path) -> None:
    data = _load_sweep(sweep.name)
    show_div = any(np.any(d["divergence_rate"] > 0) for d in data.values())
    ncols = 3 if show_div else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2))
    for f in FILTERS:
        c, ls = STYLES[f]
        d = data[f]
        axes[0].plot(d["value"], d["pos_rmse"], color=c, ls=ls, marker="o", ms=4, label=f)
        axes[1].plot(d["value"], d["mean_nees"], color=c, ls=ls, marker="o", ms=4, label=f)
        if show_div:
            axes[2].plot(d["value"], d["divergence_rate"], color=c, ls=ls, marker="o", ms=4, label=f)
    axes[1].axhline(EXPECTED_NEES, color="k", lw=1, label=f"expected NEES = {EXPECTED_NEES:g}")
    axes[0].set_ylabel("position RMSE [m]")
    axes[1].set_ylabel("mean NEES")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    if show_div:
        axes[2].set_ylabel("divergence rate")
        axes[2].set_ylim(-0.02, 1.02)
    ref = SWEEP_REFERENCE.get(sweep.name)
    for ax in axes:
        if sweep.log_x:
            ax.set_xscale("log")
        if ref is not None:
            ax.axvline(ref, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel(sweep.parameter)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    note = "metrics over non-diverged runs; dashed line: nominal value" if ref is not None \
        else "metrics over non-diverged runs"
    fig.suptitle(f"sweep {sweep.name} (N = 100): {note}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("nominal_example_trajectory.png", plot_example),
                     ("nominal_error_over_time.png", plot_error_over_time)):
        fn(FIG_DIR / name)
        print(f"wrote {FIG_DIR / name}")
    for sweep in SWEEPS:
        if (RESULTS_DIR / f"sweep_{sweep.name}.csv").exists():
            path = FIG_DIR / f"sweep_{sweep.name}.png"
            plot_sweep(sweep, path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
