"""Generate the synthetic scenario grid and summarise it with the IMU-only (strapdown) baseline.

Usage:
  python3 scripts/generate_synthetic_grid.py --out results_synth --platforms ugv uav --seeds 3
  python3 scripts/generate_synthetic_grid.py --out results_synth --export   # also write .npz files

Writes <out>/summary.csv (one row per scenario x seed) and <out>/scenarios.json.
The summary is a sanity table (does each defect change the data?), not a performance claim.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.nav_compare.baselines.strapdown import drift_metrics, run_strapdown
from experiments.nav_compare.data.scenarios import (
    generate,
    initial_state_for,
    scenario_grid,
)
from experiments.nav_compare.data.sequence import split_into_segments


def export_npz(path, seq):
    obs_t = np.array([o.t_s for o in seq.observations])
    obs_z = np.array([o.z for o in seq.observations]).reshape(-1, 3)
    obs_std = np.array([np.sqrt(o.r_meas[0, 0]) for o in seq.observations])
    np.savez_compressed(
        path,
        t=seq.imu.t_s, gyro=seq.imu.gyro_rad_s, accel=seq.imu.accel_m_s2,
        ref_t=seq.reference.t_s, ref_p=seq.reference.p_w, ref_v=seq.reference.v_w,
        ref_q=seq.reference.q_wb, obs_t=obs_t, obs_z=obs_z, obs_std=obs_std,
        outage=np.array(seq.outage_s),
    )  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_synth")
    ap.add_argument("--platforms", nargs="+", default=["ugv", "uav"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, specs = [], {}
    for platform in a.platforms:
        for name, sc in scenario_grid(platform).items():
            specs[f"{platform}/{name}"] = sc.__dict__ | {"extra": {}}
            for seed in range(a.seeds):
                seq = generate(sc, seed)
                if a.export:
                    export_npz(out / f"{seq.id}.npz", seq)
                _, gaps = split_into_segments(
                    seq.imu.t_s, seq.imu.gyro_rad_s, seq.imu.accel_m_s2
                )
                row = {"platform": platform, "scenario": name, "seed": seed,
                       "samples": len(seq.imu.t_s), "gaps": gaps,
                       "gps_updates": len(seq.observations)}  # fmt: skip
                if gaps == 0:
                    init = initial_state_for(sc, seq, seed)
                    p, _, _ = run_strapdown(seq.imu, init)
                    m = drift_metrics(seq.reference.t_s, p, seq.reference.p_w)
                    row["strapdown_final_m"] = round(m["final_position_error_3d_m"], 3)
                    row["strapdown_rmse_m"] = round(m["rmse_position_3d_m"], 3)
                rows.append(row)
    keys = list(rows[0]) + ["strapdown_final_m", "strapdown_rmse_m"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "scenarios.json").write_text(json.dumps(specs, indent=1, default=str))
    print(f"{len(rows)} rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
