"""Figures of a synthetic scenario for every motion profile.

For each profile a scenario with all error sources enabled (including IMU
outliers, measurement outliers and one dropout window) is generated and the
following PNGs are written to ``synthetic/figures/``:

* ``<profile>_trajectory.png``: true trajectory and position measurements;
* ``<profile>_gyro.png`` and ``<profile>_accel.png``: clean versus corrupted
  signals and their difference together with the true bias;
* ``<profile>_bias.png``: true gyro and accelerometer bias over time.

Usage (from the repository root)::

    python -m synthetic.plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .scenario import ScenarioConfig, generate  # noqa: E402
from .trajectories import PROFILES  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
AXES = ("x", "y", "z")


def demo_config(profile: str) -> ScenarioConfig:
    """Scenario used for the figures: default errors plus outliers and one dropout window."""
    return ScenarioConfig(
        profile=profile,
        duration=60.0,
        seed=0,
        gyro_outlier_prob=1e-3,
        accel_outlier_prob=1e-3,
        meas_outlier_prob=0.02,
        meas_dropout_windows=((20.0, 10.0),),
    )


def _shade_dropout(ax, cfg: ScenarioConfig) -> None:
    for start, duration in cfg.meas_dropout_windows:
        ax.axvspan(start, start + duration, color="C3", alpha=0.12, lw=0)


def plot_trajectory(out: dict, cfg: ScenarioConfig, path: Path) -> None:
    fig = plt.figure(figsize=(13, 5.5))
    ax_xy = fig.add_subplot(1, 2, 1)
    p, z, ok, outl = out["p_true"], out["z_meas"], out["meas_available"], out["meas_outlier_mask"]
    ax_xy.plot(p[:, 0], p[:, 1], "k-", lw=1.2, label="truth")
    ax_xy.plot(z[ok & ~outl, 0], z[ok & ~outl, 1], ".", color="C0", ms=3, label="measurement")
    ax_xy.plot(z[outl, 0], z[outl, 1], "x", color="C3", ms=6, label="measurement outlier")
    ax_xy.plot(*p[0, :2], "go", label="start")
    ax_xy.set_xlabel("east [m]")
    ax_xy.set_ylabel("north [m]")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)
    ax_xy.set_title("top view")

    for i, name in enumerate(("east", "north", "up")):
        ax = fig.add_subplot(3, 2, 2 * i + 2)
        ax.plot(out["t"], p[:, i], "k-", lw=1.2)
        ax.plot(out["t_meas"], z[:, i], ".", color="C0", ms=2)
        _shade_dropout(ax, cfg)
        ax.set_ylabel(f"{name} [m]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("position vs time (shaded: measurement dropout)")
        if i == 2:
            ax.set_xlabel("time [s]")
    fig.suptitle(f"{cfg.profile}: trajectory and position measurements")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensor(out: dict, cfg: ScenarioConfig, sensor: str, unit: str, path: Path) -> None:
    t = out["t"]
    clean, meas, bias = out[f"{sensor}_clean"], out[f"{sensor}_meas"], out[f"{sensor}_bias"]
    mask = out[f"{sensor}_outlier_mask"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 7.5), sharex=True)
    for i, axis in enumerate(AXES):
        ax = axes[i, 0]
        ax.plot(t, meas[:, i], color="C1", lw=0.6, label="corrupted")
        ax.plot(t, clean[:, i], "k-", lw=0.9, label="clean")
        ax.set_ylabel(f"{axis} [{unit}]")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        diff = meas[:, i] - clean[:, i]
        ax.plot(t[~mask[:, i]], diff[~mask[:, i]], color="C1", lw=0, marker=".", ms=1,
                label="corrupted - clean")
        ax.plot(t[mask[:, i]], diff[mask[:, i]], "x", color="C3", ms=5, label="outlier spike")
        ax.plot(t, bias[:, i], "b-", lw=1.2, label="true bias")
        # Keep the bias and noise visible; outlier spikes may extend beyond the axis limits.
        core = diff[~mask[:, i]]
        lo, hi = np.min(core), np.max(core)
        pad = 0.1 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title("clean vs corrupted")
    axes[0, 1].set_title("error (outlier spikes clipped by the axis limits)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"{cfg.profile}: {sensor}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bias(out: dict, cfg: ScenarioConfig, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, sensor, unit in zip(axes, ("gyro", "accel"), ("rad/s", "m/s$^2$")):
        for i, axis in enumerate(AXES):
            ax.plot(out["t"], out[f"{sensor}_bias"][:, i], lw=1.2, label=axis)
        ax.set_ylabel(f"{sensor} bias [{unit}]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{cfg.profile}: true bias (turn-on bias + random walk)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES:
        cfg = demo_config(profile)
        out = generate(cfg)
        targets = {
            f"{profile}_trajectory.png": lambda p: plot_trajectory(out, cfg, p),
            f"{profile}_gyro.png": lambda p: plot_sensor(out, cfg, "gyro", "rad/s", p),
            f"{profile}_accel.png": lambda p: plot_sensor(out, cfg, "accel", "m/s$^2$", p),
            f"{profile}_bias.png": lambda p: plot_bias(out, cfg, p),
        }
        for name, make in targets.items():
            make(FIG_DIR / name)
            print(f"wrote {FIG_DIR / name}")


if __name__ == "__main__":
    main()
