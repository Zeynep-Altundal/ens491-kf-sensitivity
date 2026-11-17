"""Figures of the nominal filter-family comparison.

Reads the CSVs written by :mod:`experiments.filter_family.run` and writes PNGs
to ``experiments/filter_family/figures/``. Usage (from the repository root)::

    python -m experiments.filter_family.run
    python -m experiments.filter_family.plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .run import RESULTS_DIR  # noqa: E402

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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("nominal_example_trajectory.png", plot_example),
                     ("nominal_error_over_time.png", plot_error_over_time)):
        fn(FIG_DIR / name)
        print(f"wrote {FIG_DIR / name}")


if __name__ == "__main__":
    main()
