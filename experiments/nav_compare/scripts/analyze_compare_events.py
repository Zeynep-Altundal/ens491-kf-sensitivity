"""Tables and figures for scripts/compare_filters_events.py.

python -m experiments.nav_compare.scripts.analyze_compare_events results/compare_events
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILT = ["KF-CV", "KF-IMU", "EKF", "ESKF", "IESKF", "UKF", "InEKF"]
COLORS = {
    "KF-CV": "#8c8c8c",
    "KF-IMU": "#2a9d8f",
    "EKF": "#e76f51",
    "ESKF": "#264653",
    "IESKF": "#e9c46a",
    "UKF": "#9b5de5",
    "InEKF": "#f15bb5",
}
SET_ORDER = ["clean", "gyro_noise_burst", "accel_noise_burst", "gyro_spikes", "accel_spikes",
             "gyro_bias_step_x", "gyro_bias_step_y", "gyro_bias_step_z", "accel_bias_step_x",
             "accel_bias_step_y", "accel_bias_step_z", "pos_noise_burst", "pos_spikes",
             "pos_bias_step", "pos_outage_10s", "pos_outage_30s", "gnss_denied",
             "imu_and_pos_burst"]  # fmt: skip


def main(d):
    d = Path(d)
    df = pd.read_csv(d / "runs.csv")
    g = df.groupby(["platform", "set", "filter"])
    agg = g.agg(rmse_p=("rmse_p", "mean"), max_p=("max_p", "mean"), nees_p=("nees_p", "mean"),
                nees_att=("nees_att", "mean"), rmse_att=("rmse_att_deg", "mean"),
                peak=("peak_err", "mean"), pre=("pre_err", "mean"),
                recovery=("recovery_s", "median"),
                not_recovered=("recovery_s", lambda s: int(s.isna().sum())))  # fmt: skip
    agg = agg.reset_index()
    agg.to_csv(d / "summary_by_set.csv", index=False)
    pd.set_option("display.width", 220)
    for plat in ("ugv", "uav"):
        a = agg[agg.platform == plat]
        t = a.pivot(index="set", columns="filter", values="rmse_p").reindex(SET_ORDER)[
            FILT
        ]
        pk = a.pivot(index="set", columns="filter", values="peak").reindex(SET_ORDER)[
            FILT
        ]
        rc = a.pivot(index="set", columns="filter", values="recovery").reindex(
            SET_ORDER
        )[FILT]
        ne = a.pivot(index="set", columns="filter", values="nees_p").reindex(SET_ORDER)[
            FILT
        ]
        print(f"\n=== {plat.upper()}  RMSE position [m]")
        print(t.round(2).to_string())
        print(f"--- {plat.upper()}  peak error in the event window [m]")
        print(pk.round(2).to_string())
        print(f"--- {plat.upper()}  recovery time [s] (median)")
        print(rc.round(1).to_string())
        print(f"--- {plat.upper()}  NEES position (expected 3)")
        print(ne.round(1).to_string())

    # heatmap of RMSE per platform
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    for ax, plat in zip(axes, ("ugv", "uav"), strict=True):
        a = agg[agg.platform == plat]
        t = a.pivot(index="set", columns="filter", values="rmse_p").reindex(SET_ORDER)[
            FILT
        ]
        rel = np.log2(t.values / t.values.min(axis=1, keepdims=True))
        ax.imshow(rel, cmap="Reds", vmin=0, vmax=3, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="w" if rel[i, j] > 1.8 else "k")  # fmt: skip
        ax.set_xticks(range(len(FILT)), FILT)
        ax.set_yticks(range(len(SET_ORDER)), SET_ORDER, fontsize=8)
        ax.set_title(
            f"{plat.upper()}: position RMSE [m], white = best in the row", fontsize=9
        )
    fig.tight_layout()
    fig.savefig(d / "rmse_heatmap.png", dpi=150)

    # example time series
    logs = d / "logs"
    if logs.exists():
        from experiments.nav_compare.compare.runner import errors
        from experiments.nav_compare.data.disturbances import (
            apply_events,
            standard_event_sets,
        )
        from experiments.nav_compare.data.recording import to_recording
        from experiments.nav_compare.data.scenarios import generate
        from experiments.nav_compare.scripts.compare_filters_events import base_scenario

        cases = [("uav", "pos_noise_burst"), ("uav", "accel_noise_burst"),
                 ("ugv", "accel_bias_step_x"), ("ugv", "pos_outage_10s")]  # fmt: skip
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        for ax, (plat, sname) in zip(axes.ravel(), cases, strict=True):
            seq = generate(base_scenario(plat), seed=0)
            evs = standard_event_sets(plat, seq.imu.t_s[-1])[sname]
            rec = to_recording(apply_events(seq, evs, seed=0))
            for fn in FILT:
                lg = dict(np.load(logs / f"{plat}_{sname}_s00_{fn}.npz"))
                lg["filter"] = fn
                E = errors(rec, lg)
                ax.plot(E["t"], E["err_p"], color=COLORS[fn], lw=1.3, label=fn)
            for e in rec["events"]:
                ax.axvspan(e["t_start"], e["t_end"], color="#f4a261", alpha=0.25, lw=0)
            ax.set_title(f"{plat.upper()} · {sname} (shaded = disturbance)", fontsize=9)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("position error [m]")
            ax.set_ylim(0, None)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(d / "examples.png", dpi=150)
    print("\nsaved", d / "summary_by_set.csv")


if __name__ == "__main__":
    main(sys.argv[1])
