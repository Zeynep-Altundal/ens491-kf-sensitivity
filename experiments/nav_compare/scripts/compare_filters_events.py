"""KF-CV, KF-IMU, EKF and ESKF on the standard disturbance sets (UGV and UAV).

Base flight: scenario 'baseline' without the default outage; each disturbance set of
experiments.nav_compare.data.disturbances.standard_event_sets is applied to the same flight (same seed), so the motion
is identical and only the disturbance changes. Writes one CSV row per run; with --logs also the
run logs for later peak analysis.

    python -m experiments.nav_compare.scripts.compare_filters_events --seeds 5 --out results/compare_events
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.nav_compare.compare.filters import FILTERS
from experiments.nav_compare.compare.runner import run_filter, summarize
from experiments.nav_compare.data.disturbances import apply_events, standard_event_sets
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid


def base_scenario(platform):
    return scenario_grid(platform)["baseline"].replace(outage_s=())


def job(args):
    platform, seed, set_name, out_dir, keep_logs = args
    seq = generate(base_scenario(platform), seed=seed)
    events = standard_event_sets(platform, seq.imu.t_s[-1])[set_name]
    rec = to_recording(apply_events(seq, events, seed=seed))
    rows = []
    for name in FILTERS:
        log = run_filter(rec, name)
        s = summarize(rec, log)
        s.update(platform=platform, seed=seed, set=set_name)
        rows.append(s)
        if keep_logs:
            p = Path(out_dir) / "logs" / f"{platform}_{set_name}_s{seed:02d}_{name}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **{k: v for k, v in log.items() if k != "filter"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/compare_events")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--logs", action="store_true")
    ap.add_argument(
        "--sets", nargs="*", default=None, help="only these disturbance sets"
    )
    ap.add_argument("--csv", default="runs.csv")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for platform in ("ugv", "uav"):
        names = a.sets or list(standard_event_sets(platform, 60.0).keys())
        for set_name in names:
            for seed in range(a.seeds):
                jobs.append((platform, seed, set_name, str(out), a.logs))
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            rows.extend(r)
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
