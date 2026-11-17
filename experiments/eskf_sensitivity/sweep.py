"""One-parameter-at-a-time sensitivity study of the ESKF.

Implements the protocol in ``experiments/eskf_sensitivity/README.md``
("Sensitivity study protocol"). Writes ``results/sweep_<name>.csv`` and
``results/sweep_dropout_timeseries.csv``.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.sweep
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synthetic.scenario import generate

from .eskf import IDX_BA, IDX_BG, ImuNoise
from .run import RESULTS_DIR, FilterSettings, initial_state, run_filter, scenario_config

N_RUNS = 30
DURATION = 60.0
EVAL_STRIDE = 20  # IMU samples between evaluation epochs (10 Hz at 200 Hz)
S_GRID = tuple(float(v) for v in np.logspace(-2, 2, 7))
GATE_PROB = 0.999
DROPOUT_START = 15.0
NOMINAL = FilterSettings()


@dataclass(frozen=True)
class Spec:
    """One setting: scenario changes and filter changes relative to the nominal case."""

    profile: str = "turning"
    scenario: tuple = ()  # ((ScenarioConfig field, value), ...)
    bias_scale: float = 1.0  # scale of the true turn-on bias standard deviations
    q_imu_scale: float = 1.0
    q_bias_scale: float = 1.0
    r_scale: float = 1.0
    p0_scale: float = 1.0
    estimate_bias: bool = True
    gate_prob: float | None = None
    dropout: tuple | None = None  # (start, duration) for the dropout metric


@dataclass(frozen=True)
class Sweep:
    name: str
    parameter: str
    grid: tuple
    variants: tuple = ("",)
    specs: dict = field(default_factory=dict)  # (value, variant) -> Spec
    log_x: bool = True
    categorical: bool = False


def _build_sweeps() -> list[Sweep]:
    sweeps = []
    for name, attr, label in (("q_imu", "q_imu_scale", "IMU white-noise Q scale s"),
                              ("q_bias", "q_bias_scale", "bias random-walk Q scale s"),
                              ("r", "r_scale", "R scale s"),
                              ("p0", "p0_scale", "P0 scale s")):
        specs = {(s, ""): Spec(**{attr: s}) for s in S_GRID}
        sweeps.append(Sweep(name, label, S_GRID, specs=specs))

    grid = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    specs = {(c, v): Spec(bias_scale=c, estimate_bias=(v == "enabled"))
             for c in grid for v in ("enabled", "disabled")}
    sweeps.append(Sweep("bias_estimation", "turn-on bias scale c", grid,
                        ("enabled", "disabled"), specs, log_x=False))

    grid = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    specs = {}
    for d in grid:
        scen = (("meas_dropout_windows", ((DROPOUT_START, d),)),) if d > 0 else ()
        specs[(d, "")] = Spec(scenario=scen, dropout=(DROPOUT_START, d) if d > 0 else None)
    sweeps.append(Sweep("dropout", "dropout duration [s]", grid, specs=specs, log_x=False))

    grid = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
    specs = {(pr, v): Spec(scenario=(("meas_outlier_prob", pr),),
                           gate_prob=GATE_PROB if v == "gate on" else None)
             for pr in grid for v in ("gate off", "gate on")}
    sweeps.append(Sweep("outliers", "measurement outlier probability", grid,
                        ("gate off", "gate on"), specs, log_x=False))

    grid = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    specs = {(pr, ""): Spec(scenario=(("gyro_outlier_prob", pr), ("accel_outlier_prob", pr)))
             for pr in grid}
    sweeps.append(Sweep("imu_spikes", "IMU spike probability per sample and axis", grid,
                        specs=specs, log_x=False))

    grid = ("straight", "turning", "aggressive")
    specs = {(pf, ""): Spec(profile=pf) for pf in grid}
    sweeps.append(Sweep("profile", "motion profile", grid, specs=specs, log_x=False,
                        categorical=True))

    grid = (1.0, 2.0, 5.0, 10.0, 20.0)
    specs = {(r, ""): Spec(scenario=(("meas_rate", r),)) for r in grid}
    sweeps.append(Sweep("meas_rate", "measurement rate [Hz]", grid, specs=specs))
    return sweeps


SWEEPS = _build_sweeps()


def run_one(spec: Spec, run_index: int) -> dict:
    """Run the filter for one setting and run index; returns per-run sums and values."""
    truth = FilterSettings(sigma_bg0=spec.bias_scale * NOMINAL.sigma_bg0,
                           sigma_ba0=spec.bias_scale * NOMINAL.sigma_ba0)
    cfg = scenario_config(spec.profile, run_index, truth, duration=DURATION, **dict(spec.scenario))
    out = generate(cfg)
    P0 = spec.p0_scale * truth.P0()
    sg, sa = np.sqrt(spec.q_imu_scale), np.sqrt(spec.q_bias_scale)
    noise = ImuNoise(cfg.gyro_noise_density * sg, cfg.accel_noise_density * sg,
                     cfg.gyro_bias_rw * sa, cfg.accel_bias_rw * sa)
    if not spec.estimate_bias:
        P0[IDX_BG, IDX_BG] = 0.0
        P0[IDX_BA, IDX_BA] = 0.0
        noise = ImuNoise(noise.gyro_noise_density, noise.accel_noise_density, 0.0, 0.0)
    R = spec.r_scale * np.diag(np.asarray(cfg.meas_sigma, dtype=float) ** 2)
    eval_index = np.arange(0, len(out["t"]), EVAL_STRIDE)
    res = {"ok": False}
    try:
        with np.errstate(all="ignore"):
            h = run_filter(out, cfg, FilterSettings(gate_prob=spec.gate_prob),
                           initial_state(out, truth, run_index), noise=noise, R=R, P0=P0,
                           eval_index=eval_index)
        if not (np.all(np.isfinite(h["err"])) and np.all(np.isfinite(h["nees_pos"]))):
            return res
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return res
    e = h["err"]
    pos2 = np.sum(e[:, 0:3] ** 2, axis=1)
    acc = h["accepted"]
    avail = h["available"]
    res.update({
        "ok": True,
        "n": len(e),
        "pos2": float(pos2.sum()),
        "vel2": float(np.sum(e[:, 3:6] ** 2)),
        "att2": float(np.sum(e[:, 6:9] ** 2)),
        "nees": float(h["nees_pos"].sum()),
        "nis": float(np.sum(h["nis"][acc])),
        "n_acc": int(acc.sum()),
        "n_rejected": int(np.sum(avail & ~acc)),
        "n_avail": int(avail.sum()),
        "bg_final": float(np.linalg.norm(e[-1, 9:12])),
        "ba_final": float(np.linalg.norm(e[-1, 12:15])),
        "t": h["t"],
        "pos2_t": pos2,
        "trP_t": np.trace(h["P_pos"], axis1=1, axis2=2),
    })
    if spec.dropout is not None:
        start, dur = spec.dropout
        inside = (h["t"] >= start) & (h["t"] < start + dur)
        res["dropout_max"] = float(np.sqrt(pos2[inside].max()))
    return res


def _task(args):
    key, spec, i = args
    return key, run_one(spec, i)


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    out = {"n_ok": len(ok), "n_failed": len(results) - len(ok)}
    if not ok:
        return out
    n = sum(r["n"] for r in ok)
    n_acc = sum(r["n_acc"] for r in ok)
    n_avail = sum(r["n_avail"] for r in ok)
    out.update({
        "pos_rmse": np.sqrt(sum(r["pos2"] for r in ok) / n),
        "vel_rmse": np.sqrt(sum(r["vel2"] for r in ok) / n),
        "att_rmse_deg": np.rad2deg(np.sqrt(sum(r["att2"] for r in ok) / n)),
        "mean_nees_pos": sum(r["nees"] for r in ok) / n,
        "mean_nis": sum(r["nis"] for r in ok) / n_acc if n_acc else float("nan"),
        "reject_rate": sum(r["n_rejected"] for r in ok) / n_avail if n_avail else float("nan"),
        "final_bg_err_rms": np.sqrt(np.mean([r["bg_final"] ** 2 for r in ok])),
        "final_ba_err_rms": np.sqrt(np.mean([r["ba_final"] ** 2 for r in ok])),
        "dropout_max_err_mean": (np.mean([r["dropout_max"] for r in ok])
                                 if "dropout_max" in ok[0] else float("nan")),
        "dropout_max_err_max": (np.max([r["dropout_max"] for r in ok])
                                if "dropout_max" in ok[0] else float("nan")),
    })
    out["t"] = ok[0]["t"]
    out["rms_pos_t"] = np.sqrt(np.mean([r["pos2_t"] for r in ok], axis=0))
    out["sigma3_t"] = 3.0 * np.sqrt(np.mean([r["trP_t"] for r in ok], axis=0))
    return out


METRICS = ("n_ok", "n_failed", "pos_rmse", "vel_rmse", "att_rmse_deg", "mean_nees_pos",
           "mean_nis", "reject_rate", "final_bg_err_rms", "final_ba_err_rms",
           "dropout_max_err_mean", "dropout_max_err_max")


def write_sweep_csv(sweep: Sweep, agg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["value", "variant"] + list(METRICS))
        for value in sweep.grid:
            for variant in sweep.variants:
                a = agg[(sweep.name, value, variant)]
                row = [value, variant]
                for m in METRICS:
                    v = a.get(m, float("nan"))
                    row.append(f"{v:.10g}" if isinstance(v, float) else v)
                writer.writerow(row)


def write_dropout_timeseries(sweep: Sweep, agg: dict, path: Path) -> None:
    t = agg[(sweep.name, sweep.grid[0], "")]["t"]
    cols, header = [t], ["t"]
    for d in sweep.grid:
        a = agg[(sweep.name, d, "")]
        cols += [a["rms_pos_t"], a["sigma3_t"]]
        header += [f"rms_pos_err_{d:g}s", f"three_sigma_{d:g}s"]
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header),
               comments="", fmt="%.10g")


def main() -> None:
    tasks = [((sw.name, value, variant), sw.specs[(value, variant)], i)
             for sw in SWEEPS for value in sw.grid for variant in sw.variants
             for i in range(N_RUNS)]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tasks)} runs (N = {N_RUNS} per setting) on {workers} processes")
    t0 = time.perf_counter()
    grouped: dict = {}
    with mp.Pool(workers) as pool:
        for key, res in pool.imap_unordered(_task, tasks, chunksize=4):
            grouped.setdefault(key, []).append(res)
    print(f"wall-clock time {time.perf_counter() - t0:.1f} s")
    agg = {key: aggregate(results) for key, results in grouped.items()}
    for sw in SWEEPS:
        path = RESULTS_DIR / f"sweep_{sw.name}.csv"
        write_sweep_csv(sw, agg, path)
        print(f"wrote {path}")
        if sw.name == "dropout":
            ts = RESULTS_DIR / "sweep_dropout_timeseries.csv"
            write_dropout_timeseries(sw, agg, ts)
            print(f"wrote {ts}")


if __name__ == "__main__":
    main()
