"""Characterisation of the KITTI raw and EuRoC IMU data (protocol in ``README.md``).

The datasets are read from their local locations (read-only); nothing is written there and no raw
data are copied into the repository.

Usage (from the repository root)::

    python -m analysis.dataset_char.characterise
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import kurtosis

from synthetic.trajectories import get_trajectory

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
KITTI_DIR = Path.home() / "data" / "kitti_raw" / "2011_09_30"
KITTI_SEQ = KITTI_DIR / "2011_09_30_drive_0016_extract" / "oxts"
EUROC_DIR = Path.home() / "data" / "euroc" / "V1_01_easy" / "mav0"

G_REF = 9.81
EARTH_R = 6378137.0
STATIC_WIN_S = 1.0
STATIC_W_STD = 0.01
STATIC_F_STD = 0.05
STATIC_SPEED = 0.05
MIN_STATIC_S = 10.0
SPIKE_WIN = 11
SPIKE_K = 6.0
TURN_RATE = 0.1
AXES = "xyz"

ROWS: list[dict] = []


def record(dataset, section, quantity, value, unit="", axis="", note=""):
    ROWS.append({"dataset": dataset, "section": section, "quantity": quantity, "axis": axis,
                 "value": value, "unit": unit, "note": note})


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

def inventory() -> dict[str, list[str]]:
    out = {}
    for name, root in (("kitti", KITTI_DIR), ("euroc", EUROC_DIR.parent)):
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        out[name] = files
    return out


def load_kitti() -> dict:
    ts = [line.strip() for line in (KITTI_SEQ / "timestamps.txt").read_text().splitlines()
          if line.strip()]
    t_ns = np.array([np.datetime64(s.replace(" ", "T"), "ns").astype(np.int64) for s in ts])
    files = sorted((KITTI_SEQ / "data").glob("*.txt"))
    rows = np.array([np.array(f.read_text().split(), dtype=float) for f in files])
    if rows.shape[1] != 30:
        raise ValueError(f"unexpected number of OXTS fields: {rows.shape[1]}")
    lat, lon, alt = rows[:, 0], rows[:, 1], rows[:, 2]
    lat0, lon0 = np.deg2rad(lat[0]), np.deg2rad(lon[0])
    east = EARTH_R * np.cos(lat0) * (np.deg2rad(lon) - lon0)
    north = EARTH_R * (np.deg2rad(lat) - lat0)
    return {
        "name": "kitti", "t": (t_ns - t_ns[0]) * 1e-9, "t_ns": t_ns, "n_files": len(files),
        "n_timestamps": len(ts),
        "gyro": rows[:, 17:20], "accel": rows[:, 11:14],
        "rpy": rows[:, 3:6],
        "gt_t": (t_ns - t_ns[0]) * 1e-9,
        "gt_p": np.column_stack([east, north, alt - alt[0]]),
        "gt_v": np.column_stack([rows[:, 7], rows[:, 6], rows[:, 10]]),  # ve, vn, vu (ENU)
        "status": {"pos_accuracy": rows[:, 23], "vel_accuracy": rows[:, 24],
                   "navstat": rows[:, 25], "numsats": rows[:, 26], "posmode": rows[:, 27],
                   "velmode": rows[:, 28], "orimode": rows[:, 29]},
        "raw_rows": rows,
    }


def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", comments="#")


def load_euroc() -> dict:
    imu = _read_csv(EUROC_DIR / "imu0" / "data.csv")
    gt = _read_csv(EUROC_DIR / "state_groundtruth_estimate0" / "data.csv")
    vic = _read_csv(EUROC_DIR / "vicon0" / "data.csv")
    imu_ns = np.loadtxt(EUROC_DIR / "imu0" / "data.csv", delimiter=",", comments="#",
                        usecols=0, dtype=np.int64)
    gt_ns = np.loadtxt(EUROC_DIR / "state_groundtruth_estimate0" / "data.csv", delimiter=",",
                       comments="#", usecols=0, dtype=np.int64)
    vic_ns = np.loadtxt(EUROC_DIR / "vicon0" / "data.csv", delimiter=",", comments="#",
                        usecols=0, dtype=np.int64)
    t0 = imu_ns[0]
    return {
        "name": "euroc", "t": (imu_ns - t0) * 1e-9, "t_ns": imu_ns,
        "gyro": imu[:, 1:4], "accel": imu[:, 4:7],
        "gt_t": (gt_ns - t0) * 1e-9, "gt_ns": gt_ns, "gt_p": gt[:, 1:4], "gt_v": gt[:, 8:11],
        "gt_bg": gt[:, 11:14], "gt_ba": gt[:, 14:17],
        "vic_t": (vic_ns - t0) * 1e-9, "vic_ns": vic_ns, "vic_p": vic[:, 1:4],
        "raw_rows": imu[:, 1:],
    }


# --------------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------------

def timing(d: dict, nominal_hz: float | None, nominal_note: str) -> dict:
    name = d["name"]
    dt = np.diff(d["t_ns"]) * 1e-9
    med = float(np.median(dt))
    big = dt > 1.5 * med
    missing = int(np.sum(np.round(dt[big] / med) - 1))
    dup_rows = int(np.sum(np.all(np.diff(d["raw_rows"], axis=0) == 0.0, axis=1)))
    record(name, "timing", "samples", len(d["t_ns"]))
    record(name, "timing", "duration", float(d["t"][-1]), "s")
    record(name, "timing", "nominal_rate", nominal_hz if nominal_hz else "OPEN", "Hz", note=nominal_note)
    record(name, "timing", "measured_rate_1_over_median_dt", 1.0 / med, "Hz")
    record(name, "timing", "mean_rate", (len(dt)) / float(d["t"][-1]), "Hz")
    record(name, "timing", "dt_std", float(np.std(dt)), "s")
    record(name, "timing", "dt_min", float(dt.min()), "s")
    record(name, "timing", "dt_max", float(dt.max()), "s")
    record(name, "timing", "intervals_above_1.5_median", int(big.sum()))
    record(name, "timing", "estimated_missing_samples", missing)
    record(name, "timing", "non_monotonic_or_zero_dt", int(np.sum(dt <= 0)))
    record(name, "timing", "exact_duplicate_data_rows", dup_rows)
    return {"dt": dt, "median": med}


def clock_relation(d: dict) -> None:
    name = d["name"]
    if name == "kitti":
        record(name, "timing", "imu_gt_clock", "identical records",
               note="IMU signals and navigation solution are fields of the same OXTS records")
        return
    imu_ns = d["t_ns"]
    for label, ns in (("groundtruth", d["gt_ns"]), ("vicon", d["vic_ns"])):
        idx = np.clip(np.searchsorted(imu_ns, ns), 1, len(imu_ns) - 1)
        nearest = np.where(np.abs(imu_ns[idx] - ns) < np.abs(imu_ns[idx - 1] - ns),
                           imu_ns[idx], imu_ns[idx - 1])
        off = (ns - nearest) * 1e-9
        record(name, "timing", f"{label}_fraction_timestamps_equal_to_imu", float(np.mean(off == 0)))
        record(name, "timing", f"{label}_max_abs_offset_to_nearest_imu", float(np.max(np.abs(off))), "s")
        record(name, "timing", f"{label}_start_minus_imu_start", float((ns[0] - imu_ns[0]) * 1e-9), "s")
        record(name, "timing", f"{label}_end_minus_imu_end", float((ns[-1] - imu_ns[-1]) * 1e-9), "s")


def quantisation(d: dict) -> None:
    for sensor, unit in (("gyro", "rad/s"), ("accel", "m/s^2")):
        for j in range(3):
            u = np.unique(d[sensor][:, j])
            step = np.diff(u)
            record(d["name"], "signal", f"{sensor}_smallest_value_step", float(step[step > 0].min()),
                   unit, AXES[j], note=f"{len(u)} distinct values")


def yaw_rate_consistency(d: dict) -> None:
    if "rpy" not in d:
        return
    yaw = np.unwrap(d["rpy"][:, 2])
    t = d["t"]
    dyaw = np.gradient(yaw, t)
    wz = d["gyro"][:, 2]
    record(d["name"], "signal", "rms(wz - dyaw/dt)", float(np.sqrt(np.mean((wz - dyaw) ** 2))), "rad/s",
           note="central difference of the recorded yaw")
    record(d["name"], "signal", "corr(wz, dyaw/dt)", float(np.corrcoef(wz, dyaw)[0, 1]))
    record(d["name"], "signal", "rms(wz)", float(np.sqrt(np.mean(wz**2))), "rad/s")


def static_segment(d: dict, rate: float) -> slice | None:
    n = len(d["t"])
    w = int(round(STATIC_WIN_S * rate))
    wn = np.linalg.norm(d["gyro"], axis=1)
    fn = np.linalg.norm(d["accel"], axis=1)

    def rolling_std(x):
        c1 = np.cumsum(np.concatenate([[0.0], x]))
        c2 = np.cumsum(np.concatenate([[0.0], x * x]))
        h = w // 2
        lo = np.clip(np.arange(n) - h, 0, n)
        hi = np.clip(np.arange(n) + h + 1, 0, n)
        cnt = hi - lo
        m = (c1[hi] - c1[lo]) / cnt
        return np.sqrt(np.maximum((c2[hi] - c2[lo]) / cnt - m * m, 0.0))

    sw, sf = rolling_std(wn), rolling_std(fn)
    for label, x, unit in (("gyro_norm", sw, "rad/s"), ("specific_force_norm", sf, "m/s^2")):
        k = int(np.argmin(x))
        record(d["name"], "noise", f"min_1s_window_std_of_{label}", float(x[k]), unit,
               note=f"at t = {d['t'][k]:.2f} s (descriptive; threshold unchanged)")
    static = (sw < STATIC_W_STD) & (sf < STATIC_F_STD)
    speed = np.linalg.norm(d["gt_v"], axis=1)
    sp = np.interp(d["t"], d["gt_t"], speed, left=np.nan, right=np.nan)
    has = np.isfinite(sp)
    static &= ~has | (np.where(has, sp, 0.0) < STATIC_SPEED)
    best, start = (0, 0), None
    for k in range(n + 1):
        if k < n and static[k]:
            start = k if start is None else start
        elif start is not None:
            if k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    length = (best[1] - best[0]) / rate
    record(d["name"], "noise", "static_fraction_of_samples", float(np.mean(static)))
    record(d["name"], "noise", "longest_static_segment", length, "s",
           note=f"t = {d['t'][best[0]]:.2f} .. {d['t'][max(best[1] - 1, 0)]:.2f} s" if length else "none")
    return slice(*best) if best[1] > best[0] else None


def stats(d: dict, seg: slice | None) -> None:
    name = d["name"]
    parts = [("whole", slice(None))]
    if seg is not None:
        parts.append(("static", seg))
    for part, sl in parts:
        for sensor, unit in (("gyro", "rad/s"), ("accel", "m/s^2")):
            x = d[sensor][sl]
            for j in range(3):
                v = x[:, j]
                for q, val in (("mean", v.mean()), ("std", v.std(ddof=1)), ("min", v.min()),
                               ("max", v.max()), ("excess_kurtosis", kurtosis(v))):
                    record(name, f"stats_{part}", f"{sensor}_{q}", float(val),
                           "" if q == "excess_kurtosis" else unit, AXES[j])
        fn = np.linalg.norm(d["accel"][sl], axis=1)
        record(name, f"stats_{part}", "specific_force_norm_mean", float(fn.mean()), "m/s^2",
               note=f"reference {G_REF}")
        record(name, f"stats_{part}", "specific_force_norm_std", float(fn.std(ddof=1)), "m/s^2")
        record(name, f"stats_{part}", "specific_force_norm_median", float(np.median(fn)), "m/s^2")


def allan(x: np.ndarray, rate: float, taus_n: np.ndarray) -> np.ndarray:
    dt = 1.0 / rate
    theta = np.concatenate([[0.0], np.cumsum(x) * dt])
    out = []
    N = len(theta)
    for m in taus_n:
        tau = m * dt
        d = theta[2 * m:] - 2 * theta[m:N - m] + theta[:N - 2 * m]
        out.append(np.sqrt(np.sum(d * d) / (2.0 * tau * tau * (N - 2 * m))))
    return np.array(out)


def noise(d: dict, rate: float, seg: slice | None, allan_rows: list) -> dict:
    name = d["name"]
    dt = 1.0 / rate
    res = {}
    for sensor, unit in (("gyro", "rad/s/sqrt(Hz)"), ("accel", "m/s^2/sqrt(Hz)")):
        for j in range(3):
            sd = np.std(np.diff(d[sensor][:, j])) / np.sqrt(2.0)
            record(name, "noise", f"{sensor}_density_upper_bound_from_differences",
                   float(sd * np.sqrt(dt)), unit, AXES[j],
                   note="whole sequence; includes motion and vibration")
    if seg is None or (seg.stop - seg.start) / rate < MIN_STATIC_S:
        L = 0.0 if seg is None else (seg.stop - seg.start) / rate
        record(name, "noise", "allan_deviation", "not computed", note=f"longest static segment {L:.2f} s < {MIN_STATIC_S} s")
        for sensor in ("gyro", "accel"):
            for q in ("white_noise_density", "bias_instability", "bias_random_walk"):
                record(name, "noise", f"{sensor}_{q}", "not identifiable", note="no static segment of 10 s")
        return res
    n = seg.stop - seg.start
    T = n / rate
    taus_n = np.unique(np.round(np.logspace(0, np.log10(n / 9), 40)).astype(int))
    taus = taus_n * dt
    for sensor, unit, bunit in (("gyro", "rad/s/sqrt(Hz)", "rad/s"),
                                ("accel", "m/s^2/sqrt(Hz)", "m/s^2")):
        for j in range(3):
            sig = allan(d[sensor][seg, j], rate, taus_n)
            res[(sensor, j)] = (taus, sig)
            for tau, s in zip(taus, sig):
                allan_rows.append({"dataset": name, "sensor": sensor, "axis": AXES[j],
                                   "tau_s": float(tau), "adev": float(s)})
            m = (taus_n >= 5) & (taus <= min(1.0, T / 9))
            if m.sum() >= 3:
                slope = float(np.polyfit(np.log(taus[m]), np.log(sig[m]), 1)[0])
                record(name, "noise", f"{sensor}_white_region_slope", slope, "", AXES[j],
                       note=f"tau {taus[m][0]:.3f} .. {taus[m][-1]:.3f} s")
                if -0.6 <= slope <= -0.4:
                    record(name, "noise", f"{sensor}_white_noise_density",
                           float(np.median(sig[m] * np.sqrt(taus[m]))), unit, AXES[j])
                else:
                    record(name, "noise", f"{sensor}_white_noise_density", "not identifiable",
                           axis=AXES[j], note=f"slope {slope:.2f} outside [-0.6, -0.4]")
            else:
                record(name, "noise", f"{sensor}_white_noise_density", "not identifiable",
                       axis=AXES[j], note="too few averaging times")
            k = int(np.argmin(sig))
            if 0 < k < len(sig) - 1:
                record(name, "noise", f"{sensor}_bias_instability", float(sig[k] / 0.664), bunit,
                       AXES[j], note=f"minimum at tau {taus[k]:.2f} s")
            else:
                record(name, "noise", f"{sensor}_bias_instability", "not identifiable", axis=AXES[j],
                       note=f"minimum at the edge of the valid range (tau {taus[k]:.2f} s, T/9 = {T / 9:.2f} s)")
            # +1/2 region: slope over the last third of the valid range
            tail = taus >= taus[-1] / 10
            s_tail = float(np.polyfit(np.log(taus[tail]), np.log(sig[tail]), 1)[0]) if tail.sum() >= 3 else float("nan")
            if 0.4 <= s_tail <= 0.6:
                record(name, "noise", f"{sensor}_bias_random_walk_region_slope", s_tail, "", AXES[j])
            else:
                record(name, "noise", f"{sensor}_bias_random_walk", "not identifiable", axis=AXES[j],
                       note=f"slope over the last decade of tau {s_tail:.2f}, no +1/2 region")
    return res


def spikes(d: dict) -> None:
    for sensor, unit in (("gyro", "rad/s"), ("accel", "m/s^2")):
        for j in range(3):
            x = d[sensor][:, j]
            r = x - median_filter(x, size=SPIKE_WIN, mode="nearest")
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            m = np.abs(r) > SPIKE_K * s
            record(d["name"], "spikes", f"{sensor}_robust_scale", float(s), unit, AXES[j])
            record(d["name"], "spikes", f"{sensor}_count", int(m.sum()), "", AXES[j])
            record(d["name"], "spikes", f"{sensor}_fraction", float(m.mean()), "", AXES[j])
            record(d["name"], "spikes", f"{sensor}_max_abs_residual", float(np.abs(r).max()), unit, AXES[j])
            record(d["name"], "spikes", f"{sensor}_max_abs_residual_in_scales",
                   float(np.abs(r).max() / s) if s > 0 else float("nan"), "", AXES[j])


def dynamics(name: str, t, v, w, gt_t=None, a=None) -> dict:
    """Ranges of speed, |w| and |a|; a is the finite difference of v unless given exactly."""
    tv = t if gt_t is None else gt_t
    speed = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(np.gradient(v, tv, axis=0) if a is None else a, axis=1)
    wn = np.linalg.norm(w, axis=1)
    out = {}
    for q, x, unit in (("speed", speed, "m/s"), ("angular_rate_norm", wn, "rad/s"),
                       ("kinematic_acceleration_norm", acc, "m/s^2")):
        for stat, val in (("min", x.min()), ("median", np.median(x)),
                          ("q95", np.quantile(x, 0.95)), ("max", x.max())):
            out[f"{q}_{stat}"] = float(val)
            if name in ("kitti", "euroc"):
                record(name, "dynamics", f"{q}_{stat}", float(val), unit)
    out["fraction_turning"] = float(np.mean(wn > TURN_RATE))
    if name in ("kitti", "euroc"):
        record(name, "dynamics", "fraction_time_turning", out["fraction_turning"],
               note=f"|w| > {TURN_RATE} rad/s")
    return out, (tv, speed, acc), (t, wn)


def ground_truth(d: dict, label: str, t, p, v=None) -> None:
    name = d["name"]
    dt = np.diff(t)
    med = float(np.median(dt))
    record(name, f"gt_{label}", "rate_1_over_median_dt", 1.0 / med, "Hz")
    record(name, f"gt_{label}", "samples", len(t))
    record(name, f"gt_{label}", "start_minus_imu_start", float(t[0] - d["t"][0]), "s")
    record(name, f"gt_{label}", "end_minus_imu_end", float(t[-1] - d["t"][-1]), "s")
    record(name, f"gt_{label}", "coverage_of_imu_interval",
           float((min(t[-1], d["t"][-1]) - max(t[0], d["t"][0])) / (d["t"][-1] - d["t"][0])))
    record(name, f"gt_{label}", "gaps_above_1.5_median", int(np.sum(dt > 1.5 * med)))
    record(name, f"gt_{label}", "max_dt", float(dt.max()), "s")
    jerk = np.diff(p, n=3, axis=0) / med**3
    record(name, f"gt_{label}", "rms_jerk_third_difference", float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))),
           "m/s^3", note="third difference of position / median dt^3")
    if v is not None:
        vfd = np.gradient(p, t, axis=0)
        record(name, f"gt_{label}", "rms_finite_difference_velocity_minus_given_velocity",
               float(np.sqrt(np.mean(np.sum((vfd - v) ** 2, axis=1)))), "m/s")


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def fig_timing(d, tm):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    jit = (tm["dt"] - tm["median"]) * 1e3
    axes[0].hist(jit, bins=100, log=True)
    axes[0].set_xlabel("dt - median(dt) [ms]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{d['name']}: IMU timestamp jitter (median dt {tm['median'] * 1e3:.4f} ms)")
    axes[1].plot(d["t"][1:], tm["dt"] * 1e3, lw=0.6)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("dt [ms]")
    axes[1].set_title("IMU sampling interval over time")
    fig.savefig(FIGURES_DIR / f"{d['name']}_timing.png", dpi=120)
    plt.close(fig)


def fig_hist(d):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for r, (sensor, unit) in enumerate((("gyro", "rad/s"), ("accel", "m/s^2"))):
        for j in range(3):
            v = d[sensor][:, j]
            axes[r, j].hist(v, bins=120)
            axes[r, j].set_title(f"{d['name']} {sensor} {AXES[j]}: kurtosis {kurtosis(v):.2f}", fontsize=9)
            axes[r, j].set_xlabel(unit)
    fig.savefig(FIGURES_DIR / f"{d['name']}_histograms.png", dpi=120)
    plt.close(fig)


def fig_allan(d, res):
    if not res:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, sensor, unit in ((axes[0], "gyro", "rad/s"), (axes[1], "accel", "m/s^2")):
        for j in range(3):
            taus, sig = res[(sensor, j)]
            ax.loglog(taus, sig, "o-", ms=3, label=AXES[j])
        ax.set_xlabel("tau [s]")
        ax.set_ylabel(f"Allan deviation [{unit}]")
        ax.set_title(f"{d['name']} {sensor}, static segment")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.savefig(FIGURES_DIR / f"{d['name']}_allan.png", dpi=120)
    plt.close(fig)


def fig_motion(d, vel, rate):
    (tv, speed, acc), (tw, wn) = vel, rate
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    axes[0].plot(tv, speed, lw=0.8)
    axes[0].set_ylabel("speed [m/s]")
    axes[1].plot(tw, wn, lw=0.5)
    axes[1].axhline(TURN_RATE, color="r", lw=0.8, ls="--", label=f"turning threshold {TURN_RATE} rad/s")
    axes[1].set_ylabel("|w| [rad/s]")
    axes[1].legend(fontsize=8)
    axes[2].plot(tv, acc, lw=0.5)
    axes[2].set_ylabel("|dv/dt| [m/s^2]")
    axes[2].set_xlabel("t [s] (from the first IMU sample)")
    axes[0].set_title(f"{d['name']}: motion dynamics (ground-truth velocity, gyroscope)")
    fig.savefig(FIGURES_DIR / f"{d['name']}_motion.png", dpi=120)
    plt.close(fig)


def fig_compare(dyn):
    names = list(dyn)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, q, unit in zip(axes[:3], ("speed", "angular_rate_norm", "kinematic_acceleration_norm"),
                           ("m/s", "rad/s", "m/s^2")):
        for k, nme in enumerate(names):
            r = dyn[nme]
            ax.plot([k, k], [r[f"{q}_min"], r[f"{q}_max"]], "-", color="0.6", lw=2)
            ax.plot(k, r[f"{q}_median"], "o", color="C0")
            ax.plot(k, r[f"{q}_q95"], "^", color="C3")
        ax.set_xticks(range(len(names)), names, rotation=30)
        ax.set_title(f"{q} [{unit}] (line min-max, o median, ^ 95 %)", fontsize=9)
    axes[3].bar(range(len(names)), [dyn[n]["fraction_turning"] for n in names])
    axes[3].set_xticks(range(len(names)), names, rotation=30)
    axes[3].set_title(f"fraction of time with |w| > {TURN_RATE} rad/s", fontsize=9)
    fig.savefig(FIGURES_DIR / "motion_comparison.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    for name, files in inv.items():
        record(name, "inventory", "files", len(files))
    allan_rows: list[dict] = []
    dyn = {}
    kitti, euroc = load_kitti(), load_euroc()
    record("kitti", "inventory", "oxts_data_files", kitti["n_files"])
    record("kitti", "inventory", "oxts_timestamps", kitti["n_timestamps"])
    for d, nominal, note in ((kitti, None, "no rate stated in the shipped files"),
                             (euroc, 200.0, "imu0/sensor.yaml rate_hz")):
        tm = timing(d, nominal, note)
        rate = 1.0 / tm["median"]
        clock_relation(d)
        quantisation(d)
        yaw_rate_consistency(d)
        seg = static_segment(d, rate)
        stats(d, seg)
        res = noise(d, rate, seg, allan_rows)
        spikes(d)
        out, vel, wr = dynamics(d["name"], d["t"], d["gt_v"], d["gyro"], gt_t=d["gt_t"])
        dyn[d["name"]] = out
        ground_truth(d, "navigation" if d["name"] == "kitti" else "estimate", d["gt_t"], d["gt_p"], d["gt_v"])
        if d["name"] == "euroc":
            ground_truth(d, "vicon", d["vic_t"], d["vic_p"])
            for j in range(3):
                record("euroc", "gt_estimate", "gyro_bias_in_groundtruth_mean", float(d["gt_bg"][:, j].mean()), "rad/s", AXES[j])
                record("euroc", "gt_estimate", "accel_bias_in_groundtruth_mean", float(d["gt_ba"][:, j].mean()), "m/s^2", AXES[j])
                record("euroc", "gt_estimate", "gyro_bias_in_groundtruth_range",
                       float(np.ptp(d["gt_bg"][:, j])), "rad/s", AXES[j])
                record("euroc", "gt_estimate", "accel_bias_in_groundtruth_range",
                       float(np.ptp(d["gt_ba"][:, j])), "m/s^2", AXES[j])
        else:
            for key, v in d["status"].items():
                if key in ("pos_accuracy", "vel_accuracy"):
                    record("kitti", "gt_navigation", f"{key}_min", float(v.min()))
                    record("kitti", "gt_navigation", f"{key}_max", float(v.max()))
                else:
                    vals, cnt = np.unique(v, return_counts=True)
                    record("kitti", "gt_navigation", f"{key}_values",
                           "; ".join(f"{int(a)}: {int(c)}" for a, c in zip(vals, cnt)))
        fig_timing(d, tm)
        fig_hist(d)
        fig_allan(d, res)
        fig_motion(d, vel, wr)
    syn = []
    t = np.arange(0, 60.0 + 1e-9, 1.0 / 200.0)
    for prof in ("straight", "turning", "aggressive"):
        st = get_trajectory(prof).evaluate(t)
        out, _, _ = dynamics(prof, t, st.v, st.w_b, a=st.a)
        dyn[prof] = out
        syn.append({"profile": prof, **out})
    with (RESULTS_DIR / "synthetic_profiles.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(syn[0]))
        w.writeheader()
        w.writerows(syn)
    fig_compare(dyn)
    with (RESULTS_DIR / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "section", "quantity", "axis", "value", "unit", "note"])
        w.writeheader()
        for r in ROWS:
            v = r["value"]
            w.writerow({**r, "value": f"{v:.6g}" if isinstance(v, float) else v})
    for name in ("kitti", "euroc"):
        rows = [r for r in allan_rows if r["dataset"] == name]
        if rows:
            with (RESULTS_DIR / f"allan_{name}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"{len(ROWS)} summary rows")


if __name__ == "__main__":
    main()
