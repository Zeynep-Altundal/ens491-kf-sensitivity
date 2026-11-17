"""Structural mismatch (IMU-to-body misalignment) and GNSS availability scenarios.

Implements Parts 1 and 2 of the "Structural mismatch and GNSS availability
protocol" in ``experiments/eskf_sensitivity/README.md``.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.misalignment
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
from pathlib import Path

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

from .run import RESULTS_DIR, FilterSettings, initial_state, run_filter, scenario_config

N_RUNS = 50
DURATION = 60.0
EVAL_STRIDE = 20
PROFILE = "turning"
LEVELS_DEG = (0.0, 1.0, 3.0, 5.0)
OUTAGE_START = 20.0
DRIFT_TIMES = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0)
BLOCKS = {"full": slice(0, 15), "position": slice(0, 3), "velocity": slice(3, 6),
          "attitude": slice(6, 9), "gyro_bias": slice(9, 12), "accel_bias": slice(12, 15)}

# GNSS cases: label -> (misalignment [deg], dropout windows, outage start or None)
GNSS_CASES = {
    "S0": (0.0, (), None),
    "S1": (0.0, ((OUTAGE_START, DURATION - OUTAGE_START + 0.1),), OUTAGE_START),
    "S2": (0.0, ((0.0, DURATION + 0.1),), 0.0),
    "S1 at 1 deg": (1.0, ((OUTAGE_START, DURATION - OUTAGE_START + 0.1),), OUTAGE_START),
    "S1 at 3 deg": (3.0, ((OUTAGE_START, DURATION - OUTAGE_START + 0.1),), OUTAGE_START),
    "S1 at 5 deg": (5.0, ((OUTAGE_START, DURATION - OUTAGE_START + 0.1),), OUTAGE_START),
}


def run_case(args):
    label, misalign_deg, windows, i = args
    settings = FilterSettings()
    overrides = {"duration": DURATION, "imu_misalignment_x_deg": misalign_deg}
    if windows:
        overrides["meas_dropout_windows"] = windows
    cfg = scenario_config(PROFILE, i, settings, **overrides)
    out = generate(cfg)
    eval_index = np.arange(0, len(out["t"]), EVAL_STRIDE)
    h = run_filter(out, cfg, settings, initial_state(out, settings, i), eval_index=eval_index)
    e = h["err"]
    nees = {name: np.einsum("ti,tij,tj->t", e[:, sl], np.linalg.inv(h["P"][:, sl, sl]), e[:, sl])
            for name, sl in BLOCKS.items()}
    acc = h["accepted"]
    return label, i, {
        "t": h["t"],
        "nees": nees,
        "pos_err": np.linalg.norm(e[:, 0:3], axis=1),
        "trP_pos": np.trace(h["P"][:, 0:3, 0:3], axis1=1, axis2=2),
        "att_err": e[:, 6:9],
        "pos_mse": float(np.mean(np.sum(e[:, 0:3] ** 2, axis=1))),
        "vel_mse": float(np.mean(np.sum(e[:, 3:6] ** 2, axis=1))),
        "att_mse": float(np.mean(np.sum(e[:, 6:9] ** 2, axis=1))),
        "nis": float(np.mean(h["nis"][acc])) if acc.any() else float("nan"),
        "bg_est": h["bg_est"][-1], "bg_true": out["gyro_bias"][eval_index[-1]],
        "ba_est": h["ba_est"][-1], "ba_true": out["accel_bias"][eval_index[-1]],
    }


def write_rows(rows: list[dict], path: Path) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def _runs(raw, label):
    return [raw[label][i] for i in range(N_RUNS)]


def part1(raw) -> None:
    summary, paired, bias_rows = [], [], []
    base = _runs(raw, "level 0")
    t = base[0]["t"]
    time_cols, time_header = [t], ["t"]
    for lvl in LEVELS_DEG:
        runs = _runs(raw, f"level {lvl:g}")
        row = {"misalignment_deg": lvl}
        for metric, key, scale in (("pos_rmse", "pos_mse", 1.0), ("vel_rmse", "vel_mse", 1.0),
                                   ("att_rmse_deg", "att_mse", np.rad2deg(1.0))):
            est, lo, hi = bootstrap(rms_stat, np.array([r[key] for r in runs]))
            row.update({metric: scale * est, f"{metric}_lo": scale * lo, f"{metric}_hi": scale * hi})
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
                        f"nees_{name}_frac_epochs_inside": float(np.mean((avg_t >= b_lo) & (avg_t <= b_hi)))})
            if name in ("full", "attitude"):
                time_cols.append(avg_t)
                time_header.append(f"nees_{name}_{lvl:g}deg")
        time_cols.append(np.rad2deg(np.sqrt(np.mean([np.sum(r["att_err"] ** 2, axis=1) for r in runs], axis=0))))
        time_header.append(f"att_rms_deg_{lvl:g}deg")
        summary.append(row)

        for sensor in ("bg", "ba"):
            est_m = np.mean([r[f"{sensor}_est"] for r in runs], axis=0)
            true_m = np.mean([r[f"{sensor}_true"] for r in runs], axis=0)
            err = np.array([r[f"{sensor}_est"] - r[f"{sensor}_true"] for r in runs])
            err0 = np.array([r[f"{sensor}_est"] - r[f"{sensor}_true"] for r in base])
            for ax in range(3):
                e_est, e_lo, e_hi = bootstrap(mean_stat, err[:, ax])
                b = {"misalignment_deg": lvl, "bias": sensor, "axis": "xyz"[ax],
                     "mean_estimate": float(est_m[ax]), "mean_true": float(true_m[ax]),
                     "mean_error": e_est, "mean_error_lo": e_lo, "mean_error_hi": e_hi}
                if lvl != 0.0:
                    d, dlo, dhi = paired_difference(mean_stat, err[:, ax], err0[:, ax])
                    b.update({"error_minus_0deg": d, "diff_lo": dlo, "diff_hi": dhi,
                              "supported": supported_difference(dlo, dhi)})
                bias_rows.append(b)
        if lvl != 0.0:
            for metric, key, stat in (("pos_rmse", "pos_mse", rms_stat), ("att_rmse_deg", "att_mse", rms_stat),
                                      ("mean_nis", "nis", mean_stat)):
                a = np.array([r[key] for r in runs])
                b0 = np.array([r[key] for r in base])
                d, lo, hi = paired_difference(stat, a, b0)
                scale = np.rad2deg(1.0) if metric == "att_rmse_deg" else 1.0
                paired.append({"misalignment_deg": lvl, "metric": metric, "minus_0deg": scale * d,
                               "ci_low": scale * lo, "ci_high": scale * hi,
                               "supported": supported_difference(lo, hi)})
            a = np.array([r["nees"]["full"].mean() for r in runs])
            b0 = np.array([r["nees"]["full"].mean() for r in base])
            d, lo, hi = paired_difference(mean_stat, a, b0)
            paired.append({"misalignment_deg": lvl, "metric": "nees_full", "minus_0deg": d,
                           "ci_low": lo, "ci_high": hi, "supported": supported_difference(lo, hi)})

    write_rows(summary, RESULTS_DIR / "misalignment_summary.csv")
    write_rows(paired, RESULTS_DIR / "misalignment_paired.csv")
    write_rows(bias_rows, RESULTS_DIR / "misalignment_bias.csv")
    np.savetxt(RESULTS_DIR / "misalignment_time.csv", np.column_stack(time_cols), delimiter=",",
               header=",".join(time_header), comments="", fmt="%.10g")

    print("== Part 1: misalignment (turning, N = 50) ==")
    for r in summary:
        print(f"{r['misalignment_deg']:g} deg: pos RMSE {r['pos_rmse']:.3f} [{r['pos_rmse_lo']:.3f}, {r['pos_rmse_hi']:.3f}] m, "
              f"vel RMSE {r['vel_rmse']:.3f} [{r['vel_rmse_lo']:.3f}, {r['vel_rmse_hi']:.3f}] m/s, "
              f"att RMSE {r['att_rmse_deg']:.3f} [{r['att_rmse_deg_lo']:.3f}, {r['att_rmse_deg_hi']:.3f}] deg, "
              f"NIS {r['mean_nis']:.3f} [{r['mean_nis_lo']:.3f}, {r['mean_nis_hi']:.3f}]")
        for name, sl in BLOCKS.items():
            print(f"     NEES {name:10} {r[f'nees_{name}']:9.3f} [{r[f'nees_{name}_lo']:.3f}, {r[f'nees_{name}_hi']:.3f}]"
                  f"  chi2 interval [{r[f'nees_{name}_chi2_lo']:.3f}, {r[f'nees_{name}_chi2_hi']:.3f}]"
                  f"  epochs inside {r[f'nees_{name}_frac_epochs_inside']:.3f}")
    print("-- paired differences to 0 deg --")
    for r in paired:
        print(f"{r['misalignment_deg']:g} deg {r['metric']:13} {r['minus_0deg']:+10.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] supported: {r['supported']}")
    print("-- final biases (t = 60 s), mean over runs --")
    for r in bias_rows:
        extra = (f"  error - 0deg {r['error_minus_0deg']:+.3e} [{r['diff_lo']:+.3e}, {r['diff_hi']:+.3e}] "
                 f"supported: {r['supported']}") if "supported" in r else ""
        print(f"{r['misalignment_deg']:g} deg {r['bias']} {r['axis']}: est {r['mean_estimate']:+.3e}, "
              f"true {r['mean_true']:+.3e}, error {r['mean_error']:+.3e} "
              f"[{r['mean_error_lo']:+.3e}, {r['mean_error_hi']:+.3e}]{extra}")


def part2(raw) -> None:
    summary, drift = [], []
    t = raw["S0"][0]["t"]
    cols, header = [t], ["t"]
    for label, (_, _, start) in GNSS_CASES.items():
        runs = _runs(raw, label)
        pe = np.array([r["pos_err"] for r in runs])
        trp = np.array([r["trP_pos"] for r in runs])
        full = np.array([r["nees"]["full"] for r in runs])
        inside = np.zeros_like(t, dtype=bool) if start is None else t >= start - 1e-9
        row = {"case": label}
        est, lo, hi = bootstrap(mean_stat, pe[:, -1])
        row.update({"final_pos_err_mean": est, "final_pos_err_lo": lo, "final_pos_err_hi": hi,
                    "final_pos_err_max": float(pe[:, -1].max())})
        est, lo, hi = bootstrap(mean_stat, pe.max(axis=1))
        row.update({"max_pos_err_mean": est, "max_pos_err_lo": lo, "max_pos_err_hi": hi,
                    "max_pos_err_max": float(pe.max())})
        for tag, mask in (("inside", inside), ("outside", ~inside)):
            if mask.any():
                est, lo, hi = bootstrap(mean_stat, full[:, mask].mean(axis=1))
                row.update({f"nees_full_{tag}": est, f"nees_full_{tag}_lo": lo, f"nees_full_{tag}_hi": hi,
                            f"epochs_{tag}": int(mask.sum())})
            else:
                row.update({f"nees_full_{tag}": float("nan"), f"epochs_{tag}": 0})
        summary.append(row)
        rms_t = np.sqrt(np.mean(pe**2, axis=0))
        sig3_t = 3.0 * np.sqrt(np.mean(trp, axis=0))
        cols += [rms_t, sig3_t]
        header += [f"rms_pos_err_{label.replace(' ', '_')}", f"three_sigma_{label.replace(' ', '_')}"]
        if start is not None:
            for tau in DRIFT_TIMES:
                k = int(np.argmin(np.abs(t - (start + tau))))
                drift.append({"case": label, "elapsed_outage_s": tau, "t": float(t[k]),
                              "rms_pos_err": float(rms_t[k]), "three_sigma": float(sig3_t[k])})
    write_rows(summary, RESULTS_DIR / "gnss_summary.csv")
    write_rows(drift, RESULTS_DIR / "gnss_drift.csv")
    np.savetxt(RESULTS_DIR / "gnss_time.csv", np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.10g")

    print("== Part 2: GNSS availability (turning, N = 50) ==")
    for r in summary:
        ins = (f"{r['nees_full_inside']:.3f} [{r['nees_full_inside_lo']:.3f}, {r['nees_full_inside_hi']:.3f}]"
               if r["epochs_inside"] else "-")
        outs = (f"{r['nees_full_outside']:.3f} [{r['nees_full_outside_lo']:.3f}, {r['nees_full_outside_hi']:.3f}]"
                if r["epochs_outside"] else "-")
        print(f"{r['case']:12} final err {r['final_pos_err_mean']:9.3f} [{r['final_pos_err_lo']:.3f}, {r['final_pos_err_hi']:.3f}] "
              f"(max {r['final_pos_err_max']:.3f}) | max err {r['max_pos_err_mean']:9.3f} [{r['max_pos_err_lo']:.3f}, "
              f"{r['max_pos_err_hi']:.3f}] (max {r['max_pos_err_max']:.3f}) | full NEES inside {ins} ({r['epochs_inside']} ep), "
              f"outside {outs} ({r['epochs_outside']} ep)")
    print("-- drift growth: RMS position error / 3-sigma at elapsed outage time --")
    for label in [c for c, v in GNSS_CASES.items() if v[2] is not None]:
        rows = [r for r in drift if r["case"] == label]
        print(f"{label:12} " + "  ".join(f"{r['elapsed_outage_s']:g}s: {r['rms_pos_err']:.2f}/{r['three_sigma']:.2f}" for r in rows))


def main() -> None:
    tasks = [(f"level {lvl:g}", lvl, (), i) for lvl in LEVELS_DEG for i in range(N_RUNS)]
    tasks += [(label, deg, win, i) for label, (deg, win, _) in GNSS_CASES.items() for i in range(N_RUNS)]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tasks)} runs on {workers} processes")
    raw: dict = {}
    with mp.Pool(workers) as pool:
        for label, i, res in pool.imap_unordered(run_case, tasks, chunksize=2):
            raw.setdefault(label, {})[i] = res
    part1(raw)
    part2(raw)


if __name__ == "__main__":
    main()
