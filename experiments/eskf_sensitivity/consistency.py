"""Full-state and sub-block NEES of the ESKF.

Implements the "Full-state consistency protocol" in
``experiments/eskf_sensitivity/README.md``.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.consistency
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os

import numpy as np
from scipy.stats import chi2

from experiments.common.bootstrap import bootstrap, mean_stat, paired_difference, supported_difference
from synthetic.scenario import generate

from .run import RESULTS_DIR, FilterSettings, initial_state, run_filter, scenario_config

N_RUNS = 50
DURATION = 60.0
EVAL_STRIDE = 20
PROFILES = ("straight", "turning", "aggressive")
BLOCKS = {"full": slice(0, 15), "position": slice(0, 3), "velocity": slice(3, 6),
          "attitude": slice(6, 9), "gyro_bias": slice(9, 12), "accel_bias": slice(12, 15)}


def run_one(args):
    profile, apply_reset, i = args
    settings = FilterSettings()
    cfg = scenario_config(profile, i, settings, duration=DURATION)
    out = generate(cfg)
    h = run_filter(out, cfg, settings, initial_state(out, settings, i),
                   eval_index=np.arange(0, len(out["t"]), EVAL_STRIDE), apply_reset=apply_reset)
    nees = {}
    for name, sl in BLOCKS.items():
        e = h["err"][:, sl]
        P = h["P"][:, sl, sl]
        nees[name] = np.einsum("ti,tij,tj->t", e, np.linalg.inv(P), e)
    return (profile, apply_reset), i, h["t"], nees


def main() -> None:
    tasks = [(p, r, i) for p in PROFILES for r in (True, False) for i in range(N_RUNS)]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tasks)} runs on {workers} processes")
    raw: dict = {}
    t = None
    with mp.Pool(workers) as pool:
        for key, i, t_i, nees in pool.imap_unordered(run_one, tasks, chunksize=2):
            raw.setdefault(key, {})[i] = nees
            t = t_i

    rows, paired = [], []
    time_avg: dict[str, dict[str, np.ndarray]] = {p: {} for p in PROFILES}
    for profile in PROFILES:
        for reset in (True, False):
            runs = raw[(profile, reset)]
            for name, sl in BLOCKS.items():
                dof = sl.stop - sl.start
                arr = np.array([runs[i][name] for i in range(N_RUNS)])  # (runs, epochs)
                est, lo, hi = bootstrap(mean_stat, arr.mean(axis=1))
                b_lo, b_hi = chi2.ppf([0.025, 0.975], dof * N_RUNS) / N_RUNS
                avg_t = arr.mean(axis=0)
                frac = float(np.mean((avg_t >= b_lo) & (avg_t <= b_hi)))
                rows.append({"profile": profile, "reset": reset, "block": name, "dof": dof,
                             "mean_nees": est, "ci_low": lo, "ci_high": hi,
                             "expected_inside_ci": lo <= dof <= hi,
                             "epoch_bound_low": b_lo, "epoch_bound_high": b_hi,
                             "fraction_epochs_inside": frac,
                             "fraction_epochs_above": float(np.mean(avg_t > b_hi)),
                             "fraction_epochs_below": float(np.mean(avg_t < b_lo))})
                if reset:
                    time_avg[profile][name] = avg_t
        on = np.array([raw[(profile, True)][i]["full"].mean() for i in range(N_RUNS)])
        off = np.array([raw[(profile, False)][i]["full"].mean() for i in range(N_RUNS)])
        est, lo, hi = paired_difference(mean_stat, on, off)
        paired.append({"profile": profile, "metric": "full NEES (reset on - reset off)",
                       "difference": est, "ci_low": lo, "ci_high": hi,
                       "supported": supported_difference(lo, hi)})

    with (RESULTS_DIR / "consistency_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    with (RESULTS_DIR / "consistency_reset_paired.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired[0]))
        w.writeheader()
        for r in paired:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    for profile in PROFILES:
        cols = [t] + [time_avg[profile][n] for n in BLOCKS]
        np.savetxt(RESULTS_DIR / f"consistency_time_{profile}.csv", np.column_stack(cols),
                   delimiter=",", comments="", fmt="%.10g",
                   header=",".join(["t"] + [f"nees_{n}" for n in BLOCKS]))

    print("== mean NEES [95 % CI], fraction of epochs inside / above / below the 95 % bounds ==")
    for r in rows:
        print(f"{r['profile']:10} reset {'on ' if r['reset'] else 'off'} {r['block']:10} "
              f"(exp {r['dof']:2d}) {r['mean_nees']:8.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
              f"inside CI: {str(r['expected_inside_ci']):5}  epochs in/above/below "
              f"{r['fraction_epochs_inside']:.3f}/{r['fraction_epochs_above']:.3f}/{r['fraction_epochs_below']:.3f} "
              f"(bounds [{r['epoch_bound_low']:.3f}, {r['epoch_bound_high']:.3f}])")
    print("== reset on vs off (paired, full NEES) ==")
    for r in paired:
        print(f"{r['profile']:10} {r['difference']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] "
              f"supported: {r['supported']}")


if __name__ == "__main__":
    main()
