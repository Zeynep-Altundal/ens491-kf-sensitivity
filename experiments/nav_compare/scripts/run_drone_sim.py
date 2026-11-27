"""The filters on the closed-loop quadrotor flight of simulation/rviz_drone (no ROS needed).

The 50 s flight has fixed segments (simulation/rviz_drone/README.md): ground, take-off, hover,
figure-eight, aggressive figure-eight, hover, landing. Its IMU (200 Hz) and position fixes (10 Hz,
0.5 m) come from synthetic/. Cases: "full", "outage_fig8" (no fixes 12-17 s, figure-eight) and
"outage_aggr" (no fixes 30-35 s, aggressive part). The condition of every time step is its flight
segment, overwritten by "outage" and "recovery" (10 s after the outage). Writes runs.csv and error
curves for the peak analysis:

    python -m experiments.nav_compare.scripts.run_drone_sim --seeds 3 --out results/drone_sim
    python -m experiments.nav_compare.peaks curves results/drone_sim/curves --out results/drone_sim/peaks

Needs the simulation/ folder of the zeynep/isaac-drone-sim branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.peaks import RECOVERY_S, save_curves

ROOT = Path(__file__).resolve().parents[3]
SEGMENTS = [(0, 1, "ground"), (1, 5, "take-off"), (5, 7, "hover"), (7, 27, "figure-eight"),
            (27, 39, "aggressive"), (39, 44, "hover"), (44, 49, "landing"), (49, 1e9, "ground")]  # fmt: skip
CASES = {"full": None, "outage_fig8": (12.0, 17.0), "outage_aggr": (30.0, 35.0)}
FILTERS = ["KF-CV", "EKF", "ESKF", "IESKF", "UKF", "InEKF"]


def flight(seed):
    sys.path.insert(0, str(ROOT / "simulation" / "rviz_drone"))
    try:
        from rviz_drone.flight import run
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "simulation/rviz_drone not found (branch zeynep/isaac-drone-sim)"
        ) from e
    return run(seed=seed)


def recording(fl, outage=None):
    cfg = json.loads(str(fl["config_json"]))
    t = fl["t"]
    pos_t = fl["t_meas"]
    valid = fl["meas_available"].astype(bool).copy()
    events = []
    if outage is not None:
        a, b = outage
        valid &= ~((pos_t >= a) & (pos_t < b))
        events.append({"type": "outage", "sensor": "pos", "t_start": a, "t_end": b})
    sig = np.asarray(cfg.get("meas_sigma", (0.5, 0.5, 0.5)), float)
    z = np.where(np.isfinite(fl["z_meas"]), fl["z_meas"], 0.0)
    meta = {"id": f"rviz_drone_s{cfg.get('seed', 0)}", "platform": "uav", "source": "ens491_quadrotor_sim", "version": "v1",
            "imu_sample": "instant",
            "imu_rate_hz": float(cfg.get("imu_rate", 200.0)),
            "true_noise": {"gyro_density": cfg["gyro_noise_density"],
                           "accel_density": cfg["accel_noise_density"],
                           "gyro_bias_rw": cfg["gyro_bias_rw"], "accel_bias_rw": cfg["accel_bias_rw"]},
            "scenario": {"imu_grade": "euroc"}}  # fmt: skip
    return {"imu_t": t, "imu_gyro": fl["gyro_meas"], "imu_accel": fl["accel_meas"],
            "imu_gyro_clean": fl["gyro_clean"], "imu_accel_clean": fl["accel_clean"],
            "truth_t": t, "truth_p": fl["p"], "truth_v": fl["v"], "truth_q": fl["q"],
            "truth_bg": fl["gyro_bias"], "truth_ba": fl["accel_bias"],
            "pos_t": pos_t, "pos_z": z, "pos_R": np.broadcast_to(np.diag(sig**2), (len(pos_t), 3, 3)).copy(),
            "pos_valid": valid, "meta": meta, "events": events}  # fmt: skip


def conditions(t, outage):
    lab = np.empty(len(t), dtype=object)
    for a, b, name in SEGMENTS:
        lab[(t >= a) & (t < b)] = name
    if outage is not None:
        a, b = outage
        lab[(t >= b) & (t < b + RECOVERY_S)] = "recovery"
        lab[(t >= a) & (t < b)] = "outage"
    return lab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--filters", nargs="+", default=FILTERS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(a.seeds):
        fl = flight(seed)
        for case, outage in CASES.items():
            rec = recording(fl, outage)
            curves, t = {}, None
            for f in a.filters:
                E = errors(rec, run_filter(rec, f))
                t, e = E["t"], E["err_p"]
                curves[f] = {"e": e, "nees": E["nees_p"], "nis": None}
                cond = conditions(t, outage)
                row = {"seed": seed, "case": case, "filter": f,
                       "rmse_p": float(np.sqrt(np.mean(e**2))), "max_p": float(e.max()),
                       "nees_p": float(np.mean(E["nees_p"]))}  # fmt: skip
                for seg in ("figure-eight", "aggressive", "outage", "recovery"):
                    m = cond == seg
                    row[f"rmse_{seg}"] = (
                        float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan
                    )
                rows.append(row)
                print(seed, case, f, f"rmse {row['rmse_p']:.3f} m")
            save_curves(out / "curves" / f"s{seed:02d}_{case}.npz", t, curves,
                        conditions(t, outage), platform="rviz_drone", set_name=case)  # fmt: skip
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
