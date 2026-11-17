"""Clarifications requested on the consistency study (Part 3 of the protocol).

(a) Exact definition of the "4.2 % of epochs" figure for ``turning`` and the
    errors behind the full-state, attitude and gyro-bias NEES around 10..18 s.
(b) Sensitivity of the finite-difference check of the attitude columns of F to
    the step size and to the perturbation magnitude.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.clarify
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os

import numpy as np
from scipy.stats import chi2

from synthetic.scenario import ScenarioConfig, generate

from .consistency import BLOCKS, EVAL_STRIDE, N_RUNS
from .eskf import ERROR_DIM, NominalState, difference, error_transition, inject, nominal_step
from .run import RESULTS_DIR, FilterSettings, initial_state, run_filter, scenario_config

WINDOW = (5.0, 25.0)
HIGHLIGHT = (10.0, 18.0)
FD_STEPS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
MAGNITUDES = (1e-4, 1e-3, 1e-2, 1e-1)


def _turning_run(i):
    settings = FilterSettings()
    cfg = scenario_config("turning", i, settings, duration=60.0)
    out = generate(cfg)
    h = run_filter(out, cfg, settings, initial_state(out, settings, i),
                   eval_index=np.arange(0, len(out["t"]), EVAL_STRIDE))
    nees = {name: np.einsum("ti,tij,tj->t", h["err"][:, sl], np.linalg.inv(h["P"][:, sl, sl]),
                            h["err"][:, sl]) for name, sl in BLOCKS.items()}
    var = np.diagonal(h["P"], axis1=1, axis2=2)
    return i, h["t"], nees, h["err"], var


def part_a() -> None:
    workers = max(1, (os.cpu_count() or 2) - 2)
    with mp.Pool(workers) as pool:
        res = sorted(pool.map(_turning_run, range(N_RUNS)), key=lambda r: r[0])
    t = res[0][1]
    avg = {name: np.mean([r[2][name] for r in res], axis=0) for name in BLOCKS}
    err = np.array([r[3] for r in res])
    var = np.array([r[4] for r in res])

    # The recomputed run-averaged NEES must equal the stored consistency time series.
    stored = np.genfromtxt(RESULTS_DIR / "consistency_time_turning.csv", delimiter=",", names=True)
    same = all(
        [f"{v:.10g}" for v in avg[name]] == [f"{v:.10g}" for v in stored[f"nees_{name}"]]
        for name in BLOCKS
    )
    print(f"(a) recomputed run-averaged NEES identical to consistency_time_turning.csv: {same}")

    lo, hi = chi2.ppf([0.025, 0.975], 15 * N_RUNS) / N_RUNS
    full = avg["full"]
    above, below = full > hi, full < lo
    print(f"    full-state NEES: {len(full)} evaluation epochs (t = 0, 0.1, ..., 60 s), each the average "
          f"over N = {N_RUNS} runs; 95 % bounds [{lo:.3f}, {hi:.3f}]")
    print(f"    epochs above upper bound: {above.sum()} ({above.mean():.4f}), below lower bound: "
          f"{below.sum()} ({below.mean():.4f}), outside in total: {(above | below).sum()} "
          f"({(above | below).mean():.4f})")
    for flag, name in ((above, "above"), (below, "below")):
        idx = np.nonzero(flag)[0]
        if idx.size:
            spans, start = [], idx[0]
            for a, b in zip(idx[:-1], idx[1:]):
                if b != a + 1:
                    spans.append((t[start], t[a]))
                    start = b
            spans.append((t[start], t[idx[-1]]))
            print(f"    time spans {name}: " + ", ".join(f"{s:.1f}-{e:.1f} s" for s, e in spans))

    w = (t >= WINDOW[0] - 1e-9) & (t <= WINDOW[1] + 1e-9)
    rms_att = np.rad2deg(np.sqrt(np.mean(err[:, :, 6:9] ** 2, axis=0)))
    sig_att = np.rad2deg(np.sqrt(np.mean(var[:, :, 6:9], axis=0)))
    rms_bg = np.sqrt(np.mean(err[:, :, 9:12] ** 2, axis=0))
    sig_bg = np.sqrt(np.mean(var[:, :, 9:12], axis=0))
    cols = [t, avg["full"], avg["attitude"], avg["gyro_bias"]]
    header = ["t", "nees_full", "nees_attitude", "nees_gyro_bias"]
    for k, a in enumerate("xyz"):
        cols += [rms_att[:, k], sig_att[:, k]]
        header += [f"rms_att_err_{a}_deg", f"mean_sigma_att_{a}_deg"]
    for k, a in enumerate("xyz"):
        cols += [rms_bg[:, k], sig_bg[:, k]]
        header += [f"rms_bg_err_{a}", f"mean_sigma_bg_{a}"]
    np.savetxt(RESULTS_DIR / "clarify_turning_window.csv", np.column_stack(cols)[w], delimiter=",",
               header=",".join(header), comments="", fmt="%.10g")
    h = (t >= HIGHLIGHT[0] - 1e-9) & (t <= HIGHLIGHT[1] + 1e-9)
    print(f"    in {HIGHLIGHT[0]:g}-{HIGHLIGHT[1]:g} s: max run-averaged NEES full {full[h].max():.3f}, "
          f"attitude {avg['attitude'][h].max():.3f}, gyro bias {avg['gyro_bias'][h].max():.3f}")
    for k, a in enumerate("xyz"):
        ratio_att = rms_att[h, k] / sig_att[h, k]
        ratio_bg = rms_bg[h, k] / sig_bg[h, k]
        print(f"    axis {a}: RMS attitude error / mean sigma in window: {ratio_att.min():.3f}..{ratio_att.max():.3f}"
              f" (RMS error up to {rms_att[h, k].max():.3f} deg); gyro bias: {ratio_bg.min():.3f}..{ratio_bg.max():.3f}")


def _fd_jacobian_attitude(x, imu, dt, step):
    x1 = nominal_step(x, *imu, dt)
    J = np.empty((ERROR_DIM, 3))
    for c, i in enumerate(range(6, 9)):
        d = np.zeros(ERROR_DIM)
        d[i] = step
        J[:, c] = (difference(nominal_step(inject(x, d), *imu, dt), x1)
                   - difference(nominal_step(inject(x, -d), *imu, dt), x1)) / (2.0 * step)
    return J


def part_b() -> None:
    out = generate(ScenarioConfig(profile="aggressive", duration=20.0, enable_white_noise=False,
                                  enable_bias_rw=False))
    k, dt = 2000, 1.0 / 200.0
    x = NominalState(out["p_true"][k], out["v_true"][k], out["q_true"][k],
                     np.array([1e-3, -2e-3, 1.5e-3]), np.array([0.05, -0.04, 0.03]))
    imu = (out["gyro_meas"][k], out["accel_meas"][k], out["gyro_meas"][k + 1], out["accel_meas"][k + 1])
    F = error_transition(x, *imu, dt)
    Fth = F[:, 6:9]
    eye = np.eye(ERROR_DIM)[:, 6:9]
    blocks = {"dp/dth": slice(0, 3), "dv/dth": slice(3, 6), "dth/dth": slice(6, 9)}

    rows = []
    print("(b) finite-difference step sensitivity (aggressive, t = 10 s): relative max|J - F| per block")
    for step in FD_STEPS:
        J = _fd_jacobian_attitude(x, imu, dt, step)
        row = {"fd_step": step}
        for name, sl in blocks.items():
            row[name] = float(np.max(np.abs(J[sl] - Fth[sl])) / np.max(np.abs(Fth[sl] - eye[sl])))
        row["bias_rows_max_abs"] = float(np.max(np.abs(J[9:15])))
        rows.append(row)
        print(f"    h = {step:.0e}: " + ", ".join(f"{n} {row[n]:.3e}" for n in blocks)
              + f", bias rows max|J| {row['bias_rows_max_abs']:.1e}")
    with (RESULTS_DIR / "clarify_F_step.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k2: f"{v:.10g}" for k2, v in r.items()})

    rows = []
    x1 = nominal_step(x, *imu, dt)
    print("(b) linearisation error for finite attitude perturbations: "
          "max over axes |g(d) - d - (F - I) d| / |(F - I) d|")
    for mag in MAGNITUDES:
        ratios = []
        for ax in range(3):
            d = np.zeros(ERROR_DIM)
            d[6 + ax] = mag
            g = difference(nominal_step(inject(x, d), *imu, dt), x1)
            lin = (F - np.eye(ERROR_DIM)) @ d
            ratios.append(np.linalg.norm(g - d - lin) / np.linalg.norm(lin))
        rows.append({"magnitude_rad": mag, "max_relative_error": float(max(ratios))})
        print(f"    |d| = {mag:.0e} rad: {max(ratios):.3e}")
    with (RESULTS_DIR / "clarify_F_magnitude.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k2: f"{v:.10g}" for k2, v in r.items()})


def main() -> None:
    part_a()
    part_b()


if __name__ == "__main__":
    main()
