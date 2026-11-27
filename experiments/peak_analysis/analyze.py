"""Peak tables of the estimator sweep (protocol: ``README.md``).

    .venv/bin/python -m experiments.peak_analysis.analyze      (from the repository root)

Writes ``results/peaks.csv`` (every peak), ``results/run_peaks.csv`` (one row per run) and
``results/tables.md``. Only Observed numbers; no explanations.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat
from experiments.estimators import data
from experiments.estimators.sweep import PROVISIONAL
from experiments.peak_analysis.peaks import SHOW_MIN_SEEDS, detect, onsets

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = HERE.parent / "estimators" / "results" / "runs"
RECOVERY = HERE.parent / "estimators" / "results" / "recovery.csv"
ESTIMATORS = ("KF", "EKF", "ESKF")
PHASES = ("hover", "straight", "figure_eight", "aggressive")
GROUPS = (("accel_bias_step_small", "Accelerometer bias step +0.2 m/s^2, 10 s"),
          ("gyro_bias_step_small", "Gyro bias step +0.01 rad/s, 10 s"),
          ("imu_noise_burst_x10", "IMU noise burst x10, 2 s"))


def load(scenario: str, estimator: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(RUNS / scenario / f"{estimator}_seed{seed}.npz") as z:
        return z["t"], np.linalg.norm(z["err_p"].astype(np.float64), axis=1)


def scenario_events() -> dict[str, list[tuple[str, float]]]:
    nominal = data.nominal_recording(0)
    return {s: onsets(data.scenario_recording(nominal, s)["events"]) for s in data.scenario_names()
            if s != "nominal"}


def recovery_by_run() -> dict[tuple[str, str, int], float]:
    """Largest recovery time over the events of a run; nan if an event has no end (n/a)."""
    out: dict[tuple[str, str, int], float] = {}
    with RECOVERY.open() as fh:
        for r in csv.DictReader(fh):
            key = (r["estimator"], r["scenario"], int(r["seed"]))
            v = float(r["recovery_s"])
            out[key] = v if key not in out else (float("nan") if np.isnan(out[key]) or np.isnan(v)
                                                 else max(out[key], v))
    return out


def _task(args):
    estimator, seed, events = args
    t, eps_n = load("nominal", estimator, seed)
    rows, peak_rows = [], []
    for scen, ev in events.items():
        _, eps_s = load(scen, estimator, seed)
        peaks, theta, D = detect(t, eps_s, eps_n, ev)
        att = [p for p in peaks if p.event]
        first = ev[0][1] if ev else 0.0
        row = {"estimator": estimator, "scenario": scen, "seed": seed, "theta": theta,
               "n_peaks": len(att), "n_unattributed": len(peaks) - len(att), "has_peak": int(bool(att)),
               "max_excess_after_onset": float(D[t >= first - 1e-9].max())}
        if att:
            big = max(att, key=lambda p: p.excess)
            row.update(peak_height=big.height, peak_excess=big.excess, peak_t=big.t_peak, peak_delay=big.delay,
                       peak_event=big.event, peak_duration=big.t_end - big.t_start)
        rows.append(row)
        for p in peaks:
            peak_rows.append({"estimator": estimator, "scenario": scen, "seed": seed, "t_start": p.t_start,
                              "t_end": p.t_end, "t_peak": p.t_peak, "height": p.height, "excess": p.excess,
                              "event": p.event, "onset": p.onset, "delay": p.delay})
    return rows, peak_rows


def _write(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def label(e: str) -> str:
    return f"{e} (prov.)" if e in PROVISIONAL else e


def _med_range(vals, digits=2) -> str:
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "-"
    return f"{np.median(v):.{digits}f} ({v.min():.{digits}f}-{v.max():.{digits}f})"


def _recovery_cell(vals) -> str:
    v = np.asarray(vals, float)
    if np.all(np.isnan(v)):
        return "n/a"
    nr = int(np.isinf(v).sum())
    if nr:
        return f"not rec. {nr}/{len(v)}"
    return f"{v.mean():.2f}"


def tables(rows: list[dict], rec: dict) -> str:
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["estimator"])].append(r)
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    n = len({r["seed"] for r in rows})
    L = ["# Error peaks of the estimator sweep (Observed only)", "",
         f"Peak rule, attribution and recovery: `../README.md`. {n} seeds per cell. `(prov.)`: own filter, "
         "provisional (`../../estimators/README.md`).", "",
         "## 1. Peaks per scenario and estimator", "",
         "Per cell: seeds with a peak / peaks per run (median) / largest peak: height [m], excess [m] "
         "(median, range over seeds with a peak) / delay from onset [s] (median, range) / recovery [s] "
         "(mean over seeds; \"not rec. k/10\": seeds not recovered; n/a: no end).", "",
         "| Scenario | Estimator | Seeds with peak | Peaks/run | Height [m] | Excess [m] | Delay [s] | Recovery [s] |",
         "|---|---|---|---|---|---|---|---|"]
    for s in scen:
        for e in ESTIMATORS:
            rr = by[(s, e)]
            pk = [r for r in rr if r["has_peak"]]
            recv = [rec.get((e, s, r["seed"]), float("nan")) for r in rr]
            L.append(f"| `{s}` | {label(e)} | {len(pk)}/{len(rr)} | {np.median([r['n_peaks'] for r in rr]):.0f} | "
                     f"{_med_range([r['peak_height'] for r in pk])} | {_med_range([r['peak_excess'] for r in pk])} | "
                     f"{_med_range([r['peak_delay'] for r in pk])} | {_recovery_cell(recv)} |")
    unatt = sum(r["n_unattributed"] for r in rows)
    L += ["", f"Peaks starting before every onset (unattributed), all runs: {unatt}.", "",
          "## 2. Disagreement between estimators", "",
          f"Scenarios in which at least one estimator shows a peak (at least {SHOW_MIN_SEEDS} of {n} seeds) "
          "and at least one other does not.", "",
          "| Scenario | " + " | ".join(label(e) for e in ESTIMATORS) + " | Peak in | No peak in |",
          "|---|" + "---|" * (len(ESTIMATORS) + 2)]
    agree = 0
    for s in scen:
        cnt = {e: sum(r["has_peak"] for r in by[(s, e)]) for e in ESTIMATORS}
        yes = [e for e in ESTIMATORS if cnt[e] >= SHOW_MIN_SEEDS]
        no = [e for e in ESTIMATORS if cnt[e] < SHOW_MIN_SEEDS]
        if yes and no:
            L.append(f"| `{s}` | " + " | ".join(f"{cnt[e]}/{n}" for e in ESTIMATORS) +
                     f" | {', '.join(yes)} | {', '.join(no)} |")
        else:
            agree += 1
    L += ["", f"Scenarios without disagreement: {agree} of {len(scen)}."]
    for i, (prefix, title) in enumerate(GROUPS):
        L += ["", f"## {3 if i < 2 else 4}. {title}, by onset phase" + (" and estimator" if i == 2 else ""), "",
              "Per cell: seeds with a peak; largest excess `max D` after the onset [m], mean [95 % bootstrap "
              "interval]; delay of the largest peak [s], median; recovery [s].", "",
              "| Onset phase (onset) | " + " | ".join(label(e) for e in ESTIMATORS) + " |",
              "|---|" + "---|" * len(ESTIMATORS)]
        for ph in PHASES:
            s = f"{prefix}_onset_{ph}"
            cells = []
            onset = None
            for e in ESTIMATORS:
                rr = sorted(by[(s, e)], key=lambda r: r["seed"])
                mx = np.array([r["max_excess_after_onset"] for r in rr])
                est, lo, hi = bootstrap(mean_stat, mx)
                pk = [r for r in rr if r["has_peak"]]
                d = f"{np.median([r['peak_delay'] for r in pk]):.2f}" if pk else "-"
                recv = [rec.get((e, s, r["seed"]), float("nan")) for r in rr]
                cells.append(f"{len(pk)}/{len(rr)}; {est:.3f} [{lo:.3f}, {hi:.3f}]; {d}; {_recovery_cell(recv)}")
            onset = {"hover": 5.5, "straight": 9.0, "figure_eight": 30.0, "aggressive": 52.0}[ph]
            L.append(f"| {ph} ({onset:g} s) | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.perf_counter()
    events = scenario_events()
    tasks = [(e, s, events) for e in ESTIMATORS for s in data.SEEDS]
    with mp.get_context("fork").Pool(min(len(tasks), 24)) as pool:
        res = pool.map(_task, tasks, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    peak_rows = [p for _, pp in res for p in pp]
    RESULTS.mkdir(exist_ok=True)
    _write(RESULTS / "run_peaks.csv", rows)
    _write(RESULTS / "peaks.csv", peak_rows)
    (RESULTS / "tables.md").write_text(tables(rows, recovery_by_run()))
    print(f"runs {len(rows)}, peaks {len(peak_rows)}, {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
