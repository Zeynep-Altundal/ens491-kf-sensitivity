"""Figures for the linear Kalman filter mis-tuning study.

Reads the CSVs written by :mod:`experiments.linear_kf.sweep` and writes PNGs to
``experiments/linear_kf/figures/``. Usage (from the repository root)::

    python -m experiments.linear_kf.sweep
    python -m experiments.linear_kf.plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .kf import kalman_filter  # noqa: E402
from .model import DEFAULT_DT, DEFAULT_STEPS, constant_velocity_model, simulate  # noqa: E402
from .sweep import DEFAULT_SEED, METRICS, N_TRAJ, SWEEPS, filter_matrices, load_results  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
SWEEP_LABELS = {"Q": r"$Q_f = s\,Q_{true}$", "R": r"$R_f = s\,R_{true}$", "P0": r"$P_{0,f} = s\,P_{0,true}$"}
EXAMPLE_S = (1e-3, 1.0, 1e3)
EXAMPLE_TRAJ = 0


def _col(name: str) -> int:
    return 1 + METRICS.index(name)


def plot_rmse(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(13, 3.8), sharey=True)
    for ax, sweep in zip(axes, SWEEPS):
        table = results[sweep]
        ax.loglog(table[:, 0], table[:, _col("pos_rmse")], "o-", label="position [m]")
        ax.loglog(table[:, 0], table[:, _col("vel_rmse")], "s-", label="velocity [m/s]")
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_title(SWEEP_LABELS[sweep])
        ax.set_xlabel("scale factor s")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_consistency(results: dict, n: int, m: int, path: Path) -> None:
    fig, axes = plt.subplots(2, len(SWEEPS), figsize=(13, 6.5), sharex=True)
    for j, sweep in enumerate(SWEEPS):
        table = results[sweep]
        for i, (name, expected, label) in enumerate(
            [("mean_nees", n, "mean NEES"), ("mean_nis", m, "mean NIS")]
        ):
            ax = axes[i, j]
            ax.loglog(table[:, 0], table[:, _col(name)], "o-", label=label)
            ax.axhline(expected, color="C3", ls="-", lw=1, label=f"expected = {expected}")
            ax.axvline(1.0, color="k", ls="--", lw=1)
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_ylabel(label)
                ax.legend()
        axes[0, j].set_title(SWEEP_LABELS[sweep])
        axes[1, j].set_xlabel("scale factor s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example_trajectory(path: Path) -> None:
    model = constant_velocity_model()
    x_true, y = simulate(model, DEFAULT_STEPS, N_TRAJ, np.random.default_rng(DEFAULT_SEED))
    x_true, y = x_true[EXAMPLE_TRAJ], y[EXAMPLE_TRAJ]
    t = DEFAULT_DT * np.arange(1, DEFAULT_STEPS + 1)

    fig, axes = plt.subplots(2, len(EXAMPLE_S), figsize=(14, 6.5), sharex=True)
    for j, s in enumerate(EXAMPLE_S):
        res = kalman_filter(y, **filter_matrices(model, "Q", s))
        for i, (label, unit) in enumerate([("position", "m"), ("velocity", "m/s")]):
            ax = axes[i, j]
            est = res.estimates[:, i]
            sigma3 = 3.0 * np.sqrt(res.covariances[:, i, i])
            ax.fill_between(t, est - sigma3, est + sigma3, color="C0", alpha=0.25,
                            label=r"estimate $\pm 3\sigma$")
            if i == 0:
                ax.plot(t, y[:, 0], ".", color="0.5", ms=3, label="measurement")
            ax.plot(t, x_true[1:, i], "k-", lw=1.2, label="truth")
            ax.plot(t, est, "C0-", lw=1, label="estimate")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"{label} [{unit}]")
        axes[0, j].set_title(rf"Q sweep, $s = {s:g}$")
        axes[1, j].set_xlabel("time [s]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = constant_velocity_model()
    results = load_results()
    outputs = {
        "rmse_vs_s.png": lambda p: plot_rmse(results, p),
        "nees_nis_vs_s.png": lambda p: plot_consistency(results, model.n, model.m, p),
        "example_trajectory_q_sweep.png": plot_example_trajectory,
    }
    for name, make in outputs.items():
        make(FIG_DIR / name)
        print(f"wrote {FIG_DIR / name}")


if __name__ == "__main__":
    main()
