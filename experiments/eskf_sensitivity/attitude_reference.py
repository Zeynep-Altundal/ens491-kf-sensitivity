"""Attitude NEES against the true body attitude and against the true sensor attitude.

Implements the "Attitude NEES reference protocol" in ``experiments/eskf_sensitivity/README.md``:
the misalignment runs at 0 and 5 degrees are recomputed with the code path of
``misalignment.run_case``; the attitude error of the filter, ``dtheta_B = Log(q_hat^-1 q_WB)``,
is converted to the error against the sensor attitude ``q_WS = q_WB q_BS`` by
``dtheta_S = Log(Exp(dtheta_B) q_BS)`` (same right perturbation at ``q_hat``, same covariance
block).

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.attitude_reference
"""

from __future__ import annotations

import multiprocessing as mp
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from experiments.common.bootstrap import bootstrap, mean_stat
from synthetic.rotation import quat_exp, quat_log, quat_multiply
from synthetic.scenario import generate

from .misalignment import DURATION, EVAL_STRIDE, N_RUNS, PROFILE, write_rows
from .run import RESULTS_DIR, FilterSettings, initial_state, run_filter, scenario_config

LEVELS_DEG = (0.0, 5.0)
ATT = slice(6, 9)


def sensor_frame_error(dtheta_b: np.ndarray, angle_deg: float) -> np.ndarray:
    """dtheta_S = Log(Exp(dtheta_B) ⊗ q_BS) for each row of dtheta_b (T, 3)."""
    q_bs = quat_exp(np.array([np.deg2rad(angle_deg), 0.0, 0.0]))
    return np.array([quat_log(quat_multiply(quat_exp(d), q_bs)) for d in dtheta_b])


def run_one(args):
    angle, i = args
    settings = FilterSettings()
    cfg = scenario_config(PROFILE, i, settings, duration=DURATION, imu_misalignment_x_deg=angle)
    out = generate(cfg)
    eval_index = np.arange(0, len(out["t"]), EVAL_STRIDE)
    h = run_filter(out, cfg, settings, initial_state(out, settings, i), eval_index=eval_index)
    d_b = h["err"][:, ATT]
    d_s = sensor_frame_error(d_b, angle)
    P_inv = np.linalg.inv(h["P"][:, ATT, ATT])
    return angle, i, {
        "t": h["t"],
        "nees_body": np.einsum("ti,tij,tj->t", d_b, P_inv, d_b),
        "nees_sensor": np.einsum("ti,tij,tj->t", d_s, P_inv, d_s),
        "err_body_deg": np.rad2deg(np.linalg.norm(d_b, axis=1)),
        "err_sensor_deg": np.rad2deg(np.linalg.norm(d_s, axis=1)),
    }


def main() -> None:
    tasks = [(a, i) for a in LEVELS_DEG for i in range(N_RUNS)]
    workers = max(1, (os.cpu_count() or 2) - 2)
    raw: dict = {}
    with mp.Pool(workers) as pool:
        for a, i, res in pool.imap_unordered(run_one, tasks):
            raw.setdefault(a, {})[i] = res
    reference = np.genfromtxt(RESULTS_DIR / "misalignment_time.csv", delimiter=",", names=True)
    lo, hi = chi2.ppf([0.025, 0.975], 3 * N_RUNS) / N_RUNS
    t = raw[0.0][0]["t"]
    cols, header, summary = [t], ["t"], []
    for a in LEVELS_DEG:
        runs = [raw[a][i] for i in range(N_RUNS)]
        for ref in ("body", "sensor"):
            arr = np.array([r[f"nees_{ref}"] for r in runs])
            avg = arr.mean(axis=0)
            est, blo, bhi = bootstrap(mean_stat, arr.mean(axis=1))
            err = np.array([r[f"err_{ref}_deg"] for r in runs])
            row = {"misalignment_deg": a, "reference": ref, "mean_att_nees": est,
                   "mean_att_nees_lo": blo, "mean_att_nees_hi": bhi, "chi2_lo": lo, "chi2_hi": hi,
                   "frac_epochs_inside": float(np.mean((avg >= lo) & (avg <= hi))),
                   "att_rms_err_deg": float(np.sqrt(np.mean(err**2)))}
            if ref == "body":
                ref_col = reference[f"nees_attitude_{a:g}deg"]
                row["max_abs_diff_to_misalignment_time_csv"] = float(np.max(np.abs(avg - ref_col)))
            summary.append(row)
            cols += [avg, np.sqrt(np.mean(err**2, axis=0))]
            header += [f"nees_att_{ref}_{a:g}deg", f"att_rms_err_{ref}_deg_{a:g}deg"]
    write_rows(summary, RESULTS_DIR / "attitude_reference_summary.csv")
    np.savetxt(RESULTS_DIR / "attitude_reference_time.csv", np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.10g")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, a in zip(axes, LEVELS_DEG):
        for ref, color in (("body", "C3"), ("sensor", "C0")):
            k = header.index(f"nees_att_{ref}_{a:g}deg")
            ax.plot(t, cols[k], color=color, lw=1,
                    label=f"against true {ref} attitude", ls="-" if ref == "body" else "--")
        ax.axhspan(lo, hi, color="0.85", label="95 % chi-square interval (N = 50)")
        ax.set_yscale("log")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("run-averaged attitude NEES (3 states)")
        ax.set_title(f"turning, misalignment {a:g} deg")
        ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR.parent / "figures" / "attitude_reference.png", dpi=120)
    plt.close(fig)
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
