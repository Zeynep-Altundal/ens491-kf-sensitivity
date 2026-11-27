"""Large initial attitude error: where the approximations differ (GPS 1 Hz present).

UGV and UAV, 60 s, initial attitude error 10 / 30 / 60 deg about a random axis (the filter is
told std_att = that error), biases unknown; EKF, ESKF, IESKF, UKF, InEKF; 5 seeds.

    python -m experiments.nav_compare.scripts.compare_init_attitude --out results/init_attitude
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid

FILTERS = ("EKF", "ESKF", "IESKF", "UKF", "InEKF")
ERRORS_DEG = (10.0, 30.0, 60.0)


def job(args):
    plat, seed, err_deg, fname = args
    sc = scenario_grid(plat)["baseline"].replace(outage_s=(), duration_s=60.0)
    rec = to_recording(generate(sc, seed=seed))
    log = run_filter(rec, fname, init_att_err=np.deg2rad(err_deg), init_seed=seed)
    E = errors(rec, log)
    t = E["t"]
    first10 = t <= 10.0
    conv = np.flatnonzero(E["err_att_deg"] < 1.0)
    return {"platform": plat, "seed": seed, "init_err_deg": err_deg, "filter": fname,
            "rmse_p": float(np.sqrt(np.mean(E["err_p"] ** 2))),
            "max_p": float(E["err_p"].max()),
            "rmse_p_first10s": float(np.sqrt(np.mean(E["err_p"][first10] ** 2))),
            "att_err_end_deg": float(E["err_att_deg"][-1]),
            "time_to_att_below_1deg_s": float(t[conv[0]]) if len(conv) else np.nan,
            "nees_p": float(np.mean(E["nees_p"])),
            "nees_att": float(np.mean(E["nees_att"]))}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/init_attitude")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s, e, f) for p in ("ugv", "uav") for s in range(a.seeds) for e in ERRORS_DEG
            for f in FILTERS]  # fmt: skip
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            rows.append(r)
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
