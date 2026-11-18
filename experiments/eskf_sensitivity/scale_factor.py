"""Structural mismatch: gyroscope scale-factor error.

Implements the "Gyroscope scale-factor mismatch protocol" in
``experiments/eskf_sensitivity/README.md``. The generator multiplies the clean gyro rate by
``(1 + s)``; the filter is unchanged and assumes ``s = 0``.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.scale_factor
"""

from __future__ import annotations

import multiprocessing as mp
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from experiments.common.bootstrap import (
    bootstrap,
    mean_stat,
    paired_difference,
    rms_stat,
    supported_difference,
)
from synthetic.scenario import generate

from .misalignment import BLOCKS, DRIFT_TIMES, DURATION, EVAL_STRIDE, N_RUNS, write_rows
from .run import RESULTS_DIR, FilterSettings, initial_state, run_filter, scenario_config

FIGURES_DIR = RESULTS_DIR.parent / "figures"
PROFILES = ("turning", "aggressive")
LEVELS = (0.0, 0.005, 0.01, 0.02)
OUTAGE_START = 20.0
SCENARIOS = {
    "S0": ((), None),
    "S1": (((OUTAGE_START, DURATION - OUTAGE_START + 0.1),), OUTAGE_START),
}
REF_WINDOW = 10.0  # [s] window for the rate-equivalent s * mean(w)
# Below this rate-equivalent [rad/s] the ratio is not reported (the mean rate is zero to rounding).
RATIO_MIN_RATE = 1e-9


def bias_times(scenario: str) -> dict[str, float]:
    """Times at which the gyro-bias error is reported (end of the reference window)."""
    return {"S0": {"t60": DURATION}, "S1": {"t20": OUTAGE_START, "t60": DURATION}}[scenario]


def run_case(args):
    profile, scen, s, i = args
    windows, _ = SCENARIOS[scen]
    settings = FilterSettings()
    overrides = {"duration": DURATION, "gyro_scale_factor": s}
    if windows:
        overrides["meas_dropout_windows"] = windows
    cfg = scenario_config(profile, i, settings, **overrides)
    out = generate(cfg)
    eval_index = np.arange(0, len(out["t"]), EVAL_STRIDE)
    h = run_filter(out, cfg, settings, initial_state(out, settings, i), eval_index=eval_index)
    e = h["err"]
    nees = {name: np.einsum("ti,tij,tj->t", e[:, sl], np.linalg.inv(h["P"][:, sl, sl]), e[:, sl])
            for name, sl in BLOCKS.items()}
    acc = h["accepted"]
    bias = {}
    for tag, tb in bias_times(scen).items():
        k = int(np.argmin(np.abs(h["t"] - tb)))
        imu = (out["t"] > tb - REF_WINDOW - 1e-9) & (out["t"] <= tb + 1e-9)
        bias[tag] = {"est": h["bg_est"][k], "true": out["gyro_bias"][eval_index[k]],
                     "rate_equiv": s * out["gyro_clean"][imu].mean(axis=0)}
    return profile, scen, s, i, {
        "t": h["t"], "nees": nees,
        "pos_err": np.linalg.norm(e[:, 0:3], axis=1),
        "trP_pos": np.trace(h["P"][:, 0:3, 0:3], axis1=1, axis2=2),
        "pos_mse": float(np.mean(np.sum(e[:, 0:3] ** 2, axis=1))),
        "vel_mse": float(np.mean(np.sum(e[:, 3:6] ** 2, axis=1))),
        "att_mse": float(np.mean(np.sum(e[:, 6:9] ** 2, axis=1))),
        "nis": float(np.mean(h["nis"][acc])) if acc.any() else float("nan"),
        "bias": bias,
    }


def analyse(raw):
    summary, paired, bias_rows, drift = [], [], [], []
    time_rows = {}
    for profile in PROFILES:
        for scen, (_, start) in SCENARIOS.items():
            base = [raw[(profile, scen, 0.0)][i] for i in range(N_RUNS)]
            t = base[0]["t"]
            for s in LEVELS:
                runs = [raw[(profile, scen, s)][i] for i in range(N_RUNS)]
                row = {"profile": profile, "scenario": scen, "scale_factor": s}
                for metric, key, scale in (("pos_rmse", "pos_mse", 1.0), ("vel_rmse", "vel_mse", 1.0),
                                           ("att_rmse_deg", "att_mse", np.rad2deg(1.0))):
                    est, lo, hi = bootstrap(rms_stat, np.array([r[key] for r in runs]))
                    row.update({metric: scale * est, f"{metric}_lo": scale * lo,
                                f"{metric}_hi": scale * hi})
                est, lo, hi = bootstrap(mean_stat, np.array([r["nis"] for r in runs]))
                row.update({"mean_nis": est, "mean_nis_lo": lo, "mean_nis_hi": hi})
                for name, sl in BLOCKS.items():
                    dof = sl.stop - sl.start
                    arr = np.array([r["nees"][name] for r in runs])
                    est, lo, hi = bootstrap(mean_stat, arr.mean(axis=1))
                    b_lo, b_hi = chi2.ppf([0.025, 0.975], dof * N_RUNS) / N_RUNS
                    avg_t = arr.mean(axis=0)
                    row.update({f"nees_{name}": est, f"nees_{name}_lo": lo, f"nees_{name}_hi": hi,
                                f"nees_{name}_chi2_lo": b_lo, f"nees_{name}_chi2_hi": b_hi,
                                f"nees_{name}_frac_epochs_inside":
                                    float(np.mean((avg_t >= b_lo) & (avg_t <= b_hi)))})
                    if name == "full":
                        time_rows[(profile, scen, s, "nees_full")] = avg_t
                pe = np.array([r["pos_err"] for r in runs])
                trp = np.array([r["trP_pos"] for r in runs])
                rms_t = np.sqrt(np.mean(pe**2, axis=0))
                sig3_t = 3.0 * np.sqrt(np.mean(trp, axis=0))
                time_rows[(profile, scen, s, "rms_pos")] = rms_t
                time_rows[(profile, scen, s, "three_sigma")] = sig3_t
                summary.append(row)
                if start is not None:
                    for tau in DRIFT_TIMES:
                        k = int(np.argmin(np.abs(t - (start + tau))))
                        drift.append({"profile": profile, "scenario": scen, "scale_factor": s,
                                      "elapsed_outage_s": tau, "t": float(t[k]),
                                      "rms_pos_err": float(rms_t[k]),
                                      "three_sigma": float(sig3_t[k])})
                for tag in bias_times(scen):
                    err = np.array([r["bias"][tag]["est"] - r["bias"][tag]["true"] for r in runs])
                    err0 = np.array([r["bias"][tag]["est"] - r["bias"][tag]["true"] for r in base])
                    req = np.array([r["bias"][tag]["rate_equiv"] for r in runs])
                    for ax in range(3):
                        e_est, e_lo, e_hi = bootstrap(mean_stat, err[:, ax])
                        b = {"profile": profile, "scenario": scen, "scale_factor": s, "time": tag,
                             "axis": "xyz"[ax],
                             "mean_estimate": float(np.mean([r["bias"][tag]["est"][ax] for r in runs])),
                             "mean_true": float(np.mean([r["bias"][tag]["true"][ax] for r in runs])),
                             "mean_error": e_est, "mean_error_lo": e_lo, "mean_error_hi": e_hi,
                             "rate_equivalent_s_mean_w": float(req[:, ax].mean())}
                        if s != 0.0:
                            d, dlo, dhi = paired_difference(mean_stat, err[:, ax], err0[:, ax])
                            b.update({"error_minus_s0": d, "diff_lo": dlo, "diff_hi": dhi,
                                      "supported": supported_difference(dlo, dhi),
                                      "ratio_to_rate_equivalent":
                                          d / float(req[:, ax].mean())
                                          if abs(req[:, ax].mean()) > RATIO_MIN_RATE else float("nan")})
                        bias_rows.append(b)
                if s != 0.0:
                    for metric, key, stat, scale in (
                            ("pos_rmse", "pos_mse", rms_stat, 1.0),
                            ("vel_rmse", "vel_mse", rms_stat, 1.0),
                            ("att_rmse_deg", "att_mse", rms_stat, np.rad2deg(1.0)),
                            ("mean_nis", "nis", mean_stat, 1.0)):
                        a = np.array([r[key] for r in runs])
                        b0 = np.array([r[key] for r in base])
                        d, lo, hi = paired_difference(stat, a, b0)
                        paired.append({"profile": profile, "scenario": scen, "scale_factor": s,
                                       "metric": metric, "minus_s0": scale * d,
                                       "ci_low": scale * lo, "ci_high": scale * hi,
                                       "supported": supported_difference(lo, hi)})
                    a = np.array([r["nees"]["full"].mean() for r in runs])
                    b0 = np.array([r["nees"]["full"].mean() for r in base])
                    d, lo, hi = paired_difference(mean_stat, a, b0)
                    paired.append({"profile": profile, "scenario": scen, "scale_factor": s,
                                   "metric": "nees_full", "minus_s0": d, "ci_low": lo,
                                   "ci_high": hi, "supported": supported_difference(lo, hi)})
    return summary, paired, bias_rows, drift, time_rows, t


def plot(time_rows, t) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for col, scen in enumerate(SCENARIOS):
            for s in LEVELS:
                line, = axes[0, col].plot(t, time_rows[(profile, scen, s, "rms_pos")], lw=1.2,
                                          label=f"s = {100 * s:g} %")
                axes[0, col].plot(t, time_rows[(profile, scen, s, "three_sigma")], lw=0.8, ls="--",
                                  color=line.get_color())
                axes[1, col].plot(t, time_rows[(profile, scen, s, "nees_full")], lw=1,
                                  color=line.get_color(), label=f"s = {100 * s:g} %")
            lo, hi = chi2.ppf([0.025, 0.975], 15 * N_RUNS) / N_RUNS
            axes[1, col].axhspan(lo, hi, color="0.85")
            axes[1, col].set_yscale("log")
            axes[0, col].set_yscale("log")
            axes[0, col].set_title(f"{profile}, {scen}: RMS position error (solid), "
                                   "3 sqrt(mean tr P_pos) (dashed)", fontsize=9)
            axes[1, col].set_title(f"{profile}, {scen}: run-averaged full NEES "
                                   "(band: 95 % chi-square interval)", fontsize=9)
            for ax in axes[:, col]:
                ax.set_xlabel("t [s]")
                ax.legend(fontsize=8)
                if scen == "S1":
                    ax.axvline(OUTAGE_START, color="k", lw=0.8, ls=":")
            axes[0, col].set_ylabel("position error [m]")
            axes[1, col].set_ylabel("NEES (15 states)")
        fig.savefig(FIGURES_DIR / f"scale_factor_{profile}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    tasks = [(p, sc, s, i) for p in PROFILES for sc in SCENARIOS for s in LEVELS for i in range(N_RUNS)]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tasks)} runs on {workers} processes", flush=True)
    raw: dict = {}
    with mp.Pool(workers) as pool:
        for p, sc, s, i, res in pool.imap_unordered(run_case, tasks, chunksize=2):
            raw.setdefault((p, sc, s), {})[i] = res
    summary, paired, bias_rows, drift, time_rows, t = analyse(raw)
    write_rows(summary, RESULTS_DIR / "scale_factor_summary.csv")
    write_rows(paired, RESULTS_DIR / "scale_factor_paired.csv")
    write_rows(bias_rows, RESULTS_DIR / "scale_factor_bias.csv")
    write_rows(drift, RESULTS_DIR / "scale_factor_drift.csv")
    plot(time_rows, t)
    print("done")


if __name__ == "__main__":
    main()
