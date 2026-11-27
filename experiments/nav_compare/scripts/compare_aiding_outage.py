"""GPS for a while, then no GPS: KF-CV, KF-IMU, EKF, ESKF with and without ground-vehicle aiding.

Cases (5 seeds each):
  ugv_long          6 min drive, GPS until 270 s, then 90 s without GPS
  ugv_short         2 min drive, GPS only for the first 10 s (then 110 s without GPS)
  ugv_static_short  as ugv_short, but the car stands still for the first 30 s
  uav_long          6 min flight, GPS until 270 s, then 90 s without GPS
Aiding for the UGV: none, NHC, NHC + wheel speed, NHC + wheel speed + steering yaw rate,
NHC + static calibration (ZUPT + ZARU while standing), NHC + magnetometer, all together.
Aiding for the UAV: none, magnetometer.

    python -m experiments.nav_compare.scripts.compare_aiding_outage --out results/aiding_outage
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.nav_compare.compare.filters import FILTERS
from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.disturbances import Event, apply_events
from experiments.nav_compare.data.magnetometer import simulate_heading
from experiments.nav_compare.data.odometry import simulate_odometry
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid

CASES = {
    "ugv_long": ("ugv", 360.0, 270.0),
    "ugv_short": ("ugv", 120.0, 10.0),
    "ugv_static_short": ("ugv", 120.0, 10.0),
    "uav_long": ("uav", 360.0, 270.0),
}
NHC = {"nhc_std": 1.0}
AIDING = {
    "none": {},
    "nhc": NHC,
    "nhc+speed": {**NHC, "odometry": "speed"},
    "nhc+speed+steer": {**NHC, "odometry": "both"},
    "nhc+static": {**NHC, "static": True},
    "nhc+mag": {**NHC, "magnetometer": True},
    "all": {**NHC, "odometry": "both", "static": True, "magnetometer": True},
}
UAV_AIDING = ("none", "nhc+mag")  # for the UAV only the magnetometer part is used


def jobs_for(seeds):
    out = []
    for case, (plat, _, _) in CASES.items():
        for seed in range(seeds):
            for f in FILTERS:
                if plat == "uav":
                    aids = ["none"] if f == "KF-CV" else list(UAV_AIDING)
                else:
                    aids = ["none", "all"] if f == "KF-CV" else list(AIDING)
                for a in aids:
                    out.append((case, seed, f, a))
    return out


def job(args):
    case, seed, fname, aid = args
    plat, dur, t_out = CASES[case]
    kw = {"outage_s": (), "duration_s": dur}
    if case == "ugv_static_short":
        kw["stops"] = ((0.0, 30.0), (70.0, 78.0))
    sc = scenario_grid(plat)["baseline"].replace(**kw)
    seq = generate(sc, seed=seed)
    seq = apply_events(seq, [Event("outage", "pos", t_out, dur + 1.0)], seed=seed)
    od = simulate_odometry(seq, seed=seed) if plat == "ugv" else None
    rec = to_recording(seq, odometry=od, magnetometer=simulate_heading(seq, seed=seed))
    aiding = dict(AIDING[aid])
    if plat == "uav":
        aiding.pop("nhc_std", None)  # no NHC for a drone
    log = run_filter(rec, fname, **aiding)
    E = errors(rec, log)
    m = E["t"] >= t_out
    return {"case": case, "seed": seed, "filter": fname, "aiding": aid,
            "final_p": float(E["err_p"][-1]), "max_p_outage": float(E["err_p"][m].max()),
            "rmse_p_outage": float(np.sqrt(np.mean(E["err_p"][m] ** 2))),
            "err_at_outage_start": float(E["err_p"][m][0]),
            "nees_p_outage": float(np.mean(E["nees_p"][m])),
            "err_att_deg_end": float(E["err_att_deg"][-1]) if "err_att_deg" in E
            else np.nan}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/aiding_outage")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(a.seeds)
    # longest jobs first, so the pool ends evenly
    jobs.sort(
        key=lambda j: (CASES[j[0]][1], j[2] in ("IESKF", "UKF", "EKF"), j[3]),
        reverse=True,
    )
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            rows.append(r)
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
