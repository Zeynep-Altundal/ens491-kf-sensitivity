"""Run many KalmanNet trainings from one sweep file and summarise them over seeds.

What this script does
---------------------
1. Reads a sweep file from ``sweeps/`` (JSON): a list of runs (dataset name, may contain ``*``;
   the model given to KalmanNet), the seeds and optionally the number of training steps.
   Every dataset pattern is expanded against the files in ``~/data/knet_benchmarks``.
2. Runs ``kalmannet_run`` once per (dataset, given model, seed), as separate processes, optionally
   several at the same time (``--jobs``). A run whose ``results.npz`` already exists is skipped, so an
   interrupted sweep can simply be started again. Each run writes its log to ``train.log`` in its
   output folder (``~/data/knet_outputs/kalmannet_<data>_<given>_s<seed>/``). A run whose training
   became NaN stops at once and leaves ``diverged.json`` instead; it is not retried (same seed, same
   NaN) unless ``--force`` is given, and the summary counts it.
3. Collects the results of all runs of the sweep into ``results/sweep_<name>_runs.csv`` (one row per
   run) and ``results/sweep_<name>_summary.csv`` (mean and standard deviation over seeds of the
   KalmanNet test MSE, the KF/EKF baseline with the same given model, and the gain).

Usage (from the repository root)::

    python -m experiments.knet_data.sweep main --dry-run      # list the runs, run nothing
    python -m experiments.knet_data.sweep main --jobs 2       # run them, two at a time
    python -m experiments.knet_data.sweep main --collect      # only rebuild the CSV files
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import DATA_DIR, load_dataset
from .kalmannet_run import normalize_given, run_dir

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "sweeps"
RESULTS_DIR = HERE / "results"
REPO_ROOT = HERE.parents[1]


def load_sweep(name: str) -> dict:
    path = Path(name) if name.endswith(".json") else SWEEP_DIR / f"{name}.json"
    with path.open() as fh:
        sweep = json.load(fh)
    sweep.setdefault("name", path.stem)
    return sweep


def expand(sweep: dict) -> list[dict]:
    """One job per (dataset, given model, seed). Dataset patterns are matched against DATA_DIR."""
    available = sorted(p.stem for p in DATA_DIR.glob("*.npz"))
    jobs = []
    for entry in sweep["runs"]:
        names = [n for n in available if fnmatch.fnmatch(n, entry["data"])]
        if not names:
            print(f"warning: no dataset matches {entry['data']!r} in {DATA_DIR}")
        for data in names:
            _, meta = load_dataset(DATA_DIR / f"{data}.npz")
            given = normalize_given(meta["benchmark"], entry.get("given", ""))
            for seed in entry.get("seeds", sweep.get("seeds", [0])):
                jobs.append({"data": data, "given": given, "seed": seed,
                             "steps": entry.get("steps", sweep.get("steps"))})
    return jobs


def run_job(job: dict, threads: int) -> tuple[dict, int, float]:
    """Run one job as a separate process. Returns the job, the exit code and the duration."""
    out = run_dir(job["data"], job["given"], job["seed"], job["steps"])
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "experiments.knet_data.kalmannet_run", "--data", job["data"],
           "--given", job["given"], "--seed", str(job["seed"])]
    if job["steps"]:
        cmd += ["--steps", str(job["steps"])]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    t0 = time.time()
    with (out / "train.log").open("w") as log:
        code = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return job, code, time.time() - t0


def finished(job: dict) -> bool:
    """True if the run has results or stopped because its training diverged."""
    out = run_dir(job["data"], job["given"], job["seed"], job["steps"])
    return (out / "results.npz").exists() or (out / "diverged.json").exists()


def collect(sweep: dict, jobs: list[dict]) -> tuple[Path, Path]:
    """Write the per-run and the per-setting (over seeds) CSV files of the sweep."""
    rows, diverged = [], {}
    for job in jobs:
        out = run_dir(job["data"], job["given"], job["seed"], job["steps"])
        res = out / "results.npz"
        if (out / "diverged.json").exists() and not res.exists():
            note = json.loads((out / "diverged.json").read_text())
            diverged.setdefault((note["data"], note["given_model"]), []).append(note["seed"])
        if res.exists():
            with np.load(res) as f:
                meta = json.loads(str(f["meta"]))
            meta.pop("dataset_meta", None)
            rows.append(meta)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_csv = RESULTS_DIR / f"sweep_{sweep['name']}_runs.csv"
    summary_csv = RESULTS_DIR / f"sweep_{sweep['name']}_summary.csv"
    if not rows:
        print("no finished runs yet")
        return runs_csv, summary_csv
    with runs_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["data"], r["given_model"]), []).append(r)
    summary = []
    for (data, given), rs in groups.items():
        knet = np.array([r["knet_test_db"] for r in rs])
        base = np.array([r["baseline_test_db"] for r in rs])
        summary.append({
            "data": data, "given_model": given, "n_seeds": len(rs),
            "n_diverged": len(diverged.get((data, given), [])),
            "diverged_seeds": " ".join(str(x) for x in diverged.get((data, given), [])),
            "knet_mean_db": round(float(knet.mean()), 2), "knet_std_db": round(float(knet.std()), 2),
            "baseline": rs[0]["baseline"], "baseline_db": round(float(base.mean()), 2),
            "gain_db": round(float(base.mean() - knet.mean()), 2),
            "knet_best_db": round(float(knet.min()), 2), "knet_worst_db": round(float(knet.max()), 2),
        })
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'data':32s} {'given model':24s} {'seeds':>5s} {'NaN':>4s} {'KalmanNet [dB]':>16s} "
          f"{'baseline [dB]':>14s} {'gain':>6s}")
    for s in summary:
        print(f"{s['data']:32s} {s['given_model'][:24]:24s} {s['n_seeds']:5d} {s['n_diverged']:4d} "
              f"{s['knet_mean_db']:9.2f} ± {s['knet_std_db']:4.2f} {s['baseline_db']:14.2f} {s['gain_db']:6.2f}")
    for key, seeds in diverged.items():
        if not any((s["data"], s["given_model"]) == key for s in summary):
            print(f"{key[0]:32s} {key[1][:24]:24s} all runs diverged (seeds {seeds})")
    print(f"\nwrote {runs_csv}\nwrote {summary_csv}")
    return runs_csv, summary_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep", help="name of a file in sweeps/ (without .json) or a path to a .json file")
    ap.add_argument("--jobs", type=int, default=1, help="runs at the same time (default 1)")
    ap.add_argument("--dry-run", action="store_true", help="only list the runs")
    ap.add_argument("--collect", action="store_true", help="only rebuild the CSV files from finished runs")
    ap.add_argument("--force", action="store_true", help="rerun runs that already have results")
    a = ap.parse_args()

    sweep = load_sweep(a.sweep)
    jobs = expand(sweep)
    done = [j for j in jobs if finished(j)]
    todo = jobs if a.force else [j for j in jobs if j not in done]
    print(f"sweep '{sweep['name']}': {len(jobs)} runs, {len(done)} already finished, {len(todo)} to run")
    if a.dry_run:
        for j in jobs:
            mark = "done" if j in done else "todo"
            print(f"  [{mark}] {j['data']:34s} given={j['given']:8s} seed={j['seed']} steps={j['steps'] or 'default'}")
        return
    if not a.collect and todo:
        threads = max(1, (os.cpu_count() or 2) // a.jobs)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for i, (job, code, dt) in enumerate(pool.map(lambda j: run_job(j, threads), todo), 1):
                status = {0: "ok", 3: "DIVERGED (NaN, see diverged.json)"}.get(
                    code, f"FAILED (exit {code}, see train.log)")
                print(f"[{i}/{len(todo)}] {job['data']} given={job['given']} seed={job['seed']}: "
                      f"{status}, {dt / 60:.1f} min (total {(time.time() - t0) / 60:.0f} min)")
    collect(sweep, jobs)


if __name__ == "__main__":
    main()
