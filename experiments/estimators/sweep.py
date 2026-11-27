"""Sweep: estimators x (nominal + scenarios) x measurement seeds (protocol: ``README.md``).

    .venv/bin/python -m experiments.estimators.sweep      (from the repository root)

Writes one results file per run (``results/runs/``, git-ignored) and the per-run metrics
``results/run_metrics.csv`` and ``results/recovery.csv``.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.estimators import data, metrics
from experiments.estimators.run import run, save

RESULTS = Path(__file__).resolve().parent / "results"
# Reported estimators of the sweep: KF = the companion KF-CV (adapter); EKF and ESKF = own filters
# (they pass checks 1-3; the companion EKF and ESKF are cross-checks, see README.md protocol change 6).
SWEEP = {"KF": "KF", "EKF": "own_EKF", "ESKF": "own_ESKF"}
PROVISIONAL: set[str] = set()


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(data.REPO), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _task(args) -> tuple[list[dict], list[dict], float]:
    label, seed, commit = args
    impl = SWEEP[label]
    t0 = time.perf_counter()
    nominal = data.nominal_recording(seed)
    tuning = data.tuning_from(data.nominal_recording(0)) if seed else data.tuning_from(nominal)
    labels = data.phase_labels()
    rows, rec_rows = [], []
    p_nom = None
    for scen in data.scenario_names():
        rec = data.scenario_recording(nominal, scen)
        out = run(impl, rec, tuning)
        if scen == "nominal":
            p_nom = out["p_hat"]
        meta = {"estimator": label, "implementation": impl, "provisional": label in PROVISIONAL,
                "scenario": scen, "seed": seed, "recording_id": rec["meta"]["id"], "tuning": tuning.__dict__,
                "git_commit": commit}
        save(RESULTS / "runs" / scen / f"{label}_seed{seed}.npz", out, meta)
        m = metrics.run_metrics(out, labels)
        rows.append({"estimator": label, "scenario": scen, "seed": seed, "runtime_s": out["runtime_s"], **m})
        if scen != "nominal":
            for ev, t_end in metrics.events(rec):
                r = float("nan") if t_end is None else metrics.recovery_time(out["t"], out["p_hat"], p_nom, t_end)
                rec_rows.append({"estimator": label, "scenario": scen, "seed": seed, "event": ev,
                                 "t_end": "" if t_end is None else t_end, "recovery_s": r})
    return rows, rec_rows, time.perf_counter() - t0


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    t0 = time.perf_counter()
    commit = _commit()
    data.truth_flight()
    data.phase_labels()
    tasks = [(label, seed, commit) for label in ("EKF", "ESKF", "KF") for seed in data.SEEDS]
    with mp.get_context("fork").Pool(min(len(tasks), 24)) as pool:
        res = pool.map(_task, tasks, chunksize=1)
    rows = [r for rr, _, _ in res for r in rr]
    rec_rows = [r for _, rr, _ in res for r in rr]
    RESULTS.mkdir(exist_ok=True)
    _write(RESULTS / "run_metrics.csv", rows)
    _write(RESULTS / "recovery.csv", rec_rows)
    wall = time.perf_counter() - t0
    cpu = sum(t for _, _, t in res)
    (RESULTS / "sweep_runtime.txt").write_text(
        f"runs {len(rows)}, wall {wall:.1f} s, sum of task times {cpu:.1f} s, workers 24, commit {commit}\n")
    print(f"runs {len(rows)}, wall {wall:.1f} s, sum of task times {cpu:.1f} s")


if __name__ == "__main__":
    main()
