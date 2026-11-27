"""Strapdown (IMU integration without any filter update) against the filters, GNSS denied.

Strapdown is the prediction step alone: the IMU is integrated from the initial state and nothing
corrects it. It is run here as the ESKF with no position fix at all, from the true initial state,
  * "strapdown"       with zero bias (what one has without a filter),
  * "strapdown, true bias" with the true turn-on bias (best case; the bias still wanders).
The filters get position fixes for the first 10 s, then none (GNSS denied):
  * ESKF, InEKF; for the car also ESKF + non-holonomic constraint (NHC) with sigma 0.1 and 1 m/s.

    python -m experiments.nav_compare.scripts.compare_strapdown --seeds 5 --out results/strapdown
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.disturbances import Event, apply_events
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid

DUR = {"ugv": 120.0, "uav": 60.0}
GPS_UNTIL = 10.0


def recording(platform, seed, denied_from):
    sc = scenario_grid(platform)["baseline"].replace(
        outage_s=(), duration_s=DUR[platform]
    )
    seq = generate(sc, seed=seed)
    seq = apply_events(
        seq, [Event("outage", "pos", denied_from, DUR[platform] + 1.0)], seed=seed
    )
    return to_recording(seq)


def summary(rec, log, name, platform, seed):
    E = errors(rec, log)
    t, e = E["t"], E["err_p"]
    out = {"platform": platform, "seed": seed, "method": name,
           "rmse_p": float(np.sqrt(np.mean(e**2))), "final_p": float(e[-1])}  # fmt: skip
    for s in (10.0, 30.0, 60.0):
        if t[-1] >= s:
            out[f"err_at_{int(s)}s"] = float(np.interp(s, t, e))
    return out


def job(args):
    platform, seed = args
    rows = []
    sd = recording(platform, seed, 0.0)  # no fix at all
    rows.append(summary(sd, run_filter(sd, "ESKF"), "strapdown", platform, seed))
    sd_true = dict(sd)
    sd_true["meta"] = {
        **sd["meta"],
        "init_bias": "truth",
        "init_bias_std": (1e-6, 1e-6),
    }
    rows.append(
        summary(
            sd_true, run_filter(sd_true, "ESKF"), "strapdown, true bias", platform, seed
        )
    )
    dn = recording(platform, seed, GPS_UNTIL)
    for f in ("ESKF", "InEKF"):
        rows.append(summary(dn, run_filter(dn, f), f, platform, seed))
    if platform == "ugv":
        for s in (0.1, 1.0):
            rows.append(summary(dn, run_filter(dn, "ESKF", nhc_std=s), f"ESKF + NHC {s:g} m/s",
                                platform, seed))  # fmt: skip
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s) for p in ("ugv", "uav") for s in range(a.seeds)]
    with Pool() as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out / 'runs.csv'}")


if __name__ == "__main__":
    main()
