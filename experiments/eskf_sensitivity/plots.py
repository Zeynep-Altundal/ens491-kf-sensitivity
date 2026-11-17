"""Figures of a single ESKF run and of the sensitivity sweeps.

Reads ``results/<profile>_run<i>_history.csv`` written by
:mod:`experiments.eskf_sensitivity.run` and the sweep CSVs written by
:mod:`experiments.eskf_sensitivity.sweep`, and writes PNGs to
``experiments/eskf_sensitivity/figures/``. Usage (from the repository root)::

    python -m experiments.eskf_sensitivity.run
    python -m experiments.eskf_sensitivity.sweep
    python -m experiments.eskf_sensitivity.plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from .run import DEFAULT_PROFILE, DEFAULT_RUN_INDEX, RESULTS_DIR  # noqa: E402
from .sweep import DROPOUT_START, SWEEPS  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
AXES = "xyz"


def load_history(stem: str) -> dict[str, np.ndarray]:
    table = np.genfromtxt(RESULTS_DIR / f"{stem}_history.csv", delimiter=",", names=True)
    return {name: table[name] for name in table.dtype.names}


def _vec(h: dict, prefix: str) -> np.ndarray:
    return np.column_stack([h[f"{prefix}_{a}"] for a in AXES])


def plot_trajectory(h: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    p, z, est = _vec(h, "p_true"), _vec(h, "z"), _vec(h, "p_est")
    ax.plot(z[:, 0], z[:, 1], ".", color="0.6", ms=3, label="measurement")
    ax.plot(p[:, 0], p[:, 1], "k-", lw=1.5, label="truth")
    ax.plot(est[:, 0], est[:, 1], "C0-", lw=1, label="estimate")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(h: dict, title: str, path: Path) -> None:
    t = h["t"]
    rows = [("p", "position", "m", 1.0), ("v", "velocity", "m/s", 1.0),
            ("th", "attitude", "deg", np.rad2deg(1.0))]
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    for i, (key, label, unit, scale) in enumerate(rows):
        for j, a in enumerate(AXES):
            ax = axes[i, j]
            err = scale * h[f"err_{key}_{a}"]
            s3 = 3.0 * scale * h[f"sigma_{key}_{a}"]
            ax.fill_between(t, -s3, s3, color="C0", alpha=0.2, label=r"$\pm 3\sigma$")
            ax.plot(t, err, "C3-", lw=0.8, label="error")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{label} {a}")
            if j == 0:
                ax.set_ylabel(f"[{unit}]")
            if i == 2:
                ax.set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title}: estimation error (truth - estimate) with filter 3-sigma bounds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_biases(h: dict, title: str, path: Path) -> None:
    t = h["t"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for i, (key, label, unit) in enumerate((("bg", "gyro bias", "rad/s"), ("ba", "accel bias", "m/s$^2$"))):
        for j, a in enumerate(AXES):
            ax = axes[i, j]
            est = h[f"{key}_est_{a}"]
            s3 = 3.0 * h[f"sigma_{key}_{a}"]
            ax.fill_between(t, est - s3, est + s3, color="C0", alpha=0.2, label=r"estimate $\pm 3\sigma$")
            ax.plot(t, h[f"{key}_true_{a}"], "k-", lw=1.5, label="truth")
            ax.plot(t, est, "C0-", lw=1, label="estimate")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{label} {a}")
            if j == 0:
                ax.set_ylabel(f"[{unit}]")
            if i == 1:
                ax.set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title}: bias estimates versus truth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nis(h: dict, title: str, path: Path) -> None:
    t, nis = h["t"], h["nis"]
    lo, hi = chi2.ppf([0.025, 0.975], df=3)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(t, nis, ".", ms=3, color="C0", label="NIS")
    ax.axhline(3.0, color="k", lw=1, label="expected = 3")
    ax.axhline(lo, color="C3", lw=1, ls="--", label="95 % interval, chi2(3)")
    ax.axhline(hi, color="C3", lw=1, ls="--")
    frac = np.mean((nis >= lo) & (nis <= hi))
    ax.set_title(f"{title}: NIS per update (mean {np.nanmean(nis):.3f}, "
                 f"{100 * frac:.1f} % inside the 95 % interval)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NIS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


EXPECTED_NEES = 3.0
SWEEP_REFERENCE = {"q_imu": 1.0, "q_bias": 1.0, "r": 1.0, "p0": 1.0, "bias_estimation": 1.0,
                   "meas_rate": 10.0}


def _load_sweep(name: str):
    table = np.genfromtxt(RESULTS_DIR / f"sweep_{name}.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    return table


def plot_sweep(sweep, path: Path) -> None:
    table = _load_sweep(sweep.name)
    panels = [("pos_rmse", "position RMSE [m]"), ("att_rmse_deg", "attitude RMSE [deg]"),
              ("mean_nees_pos", "mean position NEES")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, variant in enumerate(sweep.variants):
        rows = table[table["variant"] == variant] if variant else table
        if sweep.categorical:
            x = np.arange(len(sweep.grid))
        else:
            x = np.asarray(rows["value"], dtype=float)
        label = variant if variant else "ESKF"
        for ax, (key, _) in zip(axes, panels):
            ax.plot(x, np.asarray(rows[key], dtype=float), marker="o", ms=5, color=f"C{k}",
                    label=label)
    axes[2].axhline(EXPECTED_NEES, color="k", lw=1, label=f"expected = {EXPECTED_NEES:g}")
    ref = SWEEP_REFERENCE.get(sweep.name)
    for ax, (_, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xlabel(sweep.parameter)
        if sweep.categorical:
            ax.set_xticks(np.arange(len(sweep.grid)), sweep.grid)
        elif sweep.log_x:
            ax.set_xscale("log")
        if ref is not None:
            ax.axvline(ref, color="0.4", lw=0.8, ls="--")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    note = "; dashed line: nominal value" if ref is not None else ""
    fig.suptitle(f"ESKF sweep {sweep.name} (turning unless varied, N = 30){note}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dropout_timeseries(sweep, path: Path) -> None:
    d = np.genfromtxt(RESULTS_DIR / "sweep_dropout_timeseries.csv", delimiter=",", names=True)
    durations = [v for v in sweep.grid if v > 0]
    fig, axes = plt.subplots(len(durations), 1, figsize=(11, 2.3 * len(durations)), sharex=True)
    for ax, dur in zip(axes, durations):
        key = f"{dur:g}s".replace(".", "")
        names = [n for n in d.dtype.names if n.startswith("rms_pos_err_")]
        rms = d[[n for n in names if n == f"rms_pos_err_{key}"][0]]
        sig = d[f"three_sigma_{key}"]
        ax.axvspan(DROPOUT_START, DROPOUT_START + dur, color="C3", alpha=0.12, lw=0,
                   label="dropout window")
        ax.plot(d["t"], d["rms_pos_err_0s"], color="0.5", lw=1, label="no dropout (RMS error)")
        ax.plot(d["t"], rms, color="C0", lw=1.3, label="RMS position error")
        ax.plot(d["t"], sig, color="C1", lw=1.3, ls="--", label=r"$3\sqrt{\mathrm{tr}\,P_{pos}}$")
        ax.set_yscale("log")
        ax.set_ylabel("[m]")
        ax.set_title(f"dropout {dur:g} s starting at {DROPOUT_START:g} s", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("ESKF position error over time with measurement dropout (RMS over N = 30 runs; "
                 "3-sigma bound from the mean filter covariance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{DEFAULT_PROFILE}_run{DEFAULT_RUN_INDEX}"
    h = load_history(stem)
    title = f"{DEFAULT_PROFILE}, run {DEFAULT_RUN_INDEX}"
    for name, fn in (("trajectory", plot_trajectory), ("errors", plot_errors),
                     ("biases", plot_biases), ("nis", plot_nis)):
        path = FIG_DIR / f"{stem}_{name}.png"
        fn(h, title, path)
        print(f"wrote {path}")
    for sweep in SWEEPS:
        if not (RESULTS_DIR / f"sweep_{sweep.name}.csv").exists():
            continue
        path = FIG_DIR / f"sweep_{sweep.name}.png"
        plot_sweep(sweep, path)
        print(f"wrote {path}")
        if sweep.name == "dropout":
            path = FIG_DIR / "sweep_dropout_error_over_time.png"
            plot_dropout_timeseries(sweep, path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
