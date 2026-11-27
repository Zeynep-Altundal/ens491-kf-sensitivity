"""KF-CV, EKF, ESKF (and the other filters) on EuRoC sequences, with and without a position outage.

    python -m experiments.nav_compare.scripts.run_euroc ~/data/euroc/V1_01_easy --out results/euroc
    python -m experiments.nav_compare.scripts.run_euroc ~/data/euroc/MH_01_easy ~/data/euroc/V2_01_easy \
        --filters KF-CV EKF ESKF InEKF --out results/euroc

Position fixes are simulated from the ground truth (EuRoC has no GNSS): 10 Hz, 0.1 m by default.
Cases: "full" (fixes all the time) and "outage" (no fixes from 40 % to 40 % + OUTAGE_S of the
flight). Writes runs.csv (one row per sequence, case, filter) and the run logs (logs/*.npz), which
the peak analysis reads with `python -m experiments.nav_compare.peaks curves <out>/curves`.
The locked test sequences (V1_03, MH_05, V2_03) are refused by the reader.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.euroc import euroc_recording
from experiments.nav_compare.peaks import nav_conditions, save_curves

OUTAGE_S = 10.0


def case_recordings(folder, pos_rate, pos_std, seed):
    full = euroc_recording(folder, pos_rate, pos_std, seed=seed)
    dur = full["imu_t"][-1]
    a = 0.4 * dur
    return {"full": full,
            "outage": euroc_recording(folder, pos_rate, pos_std, outage=(a, a + OUTAGE_S), seed=seed)}  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sequences", nargs="+")
    ap.add_argument(
        "--filters",
        nargs="+",
        default=["KF-CV", "EKF", "ESKF", "IESKF", "UKF", "InEKF"],
    )
    ap.add_argument("--pos-rate", type=float, default=10.0)
    ap.add_argument("--pos-std", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in a.sequences:
        name = Path(seq).expanduser().name
        for case, rec in case_recordings(seq, a.pos_rate, a.pos_std, a.seed).items():
            ev = rec["events"]
            curves, t_log = {}, None
            for f in a.filters:
                log = run_filter(rec, f)
                E = errors(rec, log)
                t, e = E["t"], E["err_p"]
                t_log = t
                curves[f] = {"e": e, "nees": E["nees_p"], "nis": None}
                row = {"sequence": name, "case": case, "filter": f,
                       "rmse_p": float(np.sqrt(np.mean(e**2))), "max_p": float(e.max()),
                       "final_p": float(e[-1]), "nees_p": float(np.mean(E["nees_p"]))}  # fmt: skip
                if "err_att_deg" in E:
                    row["rmse_att_deg"] = float(np.sqrt(np.mean(E["err_att_deg"] ** 2)))
                if ev:
                    w = (t >= ev[0]["t_start"]) & (t < ev[0]["t_end"])
                    row["peak_in_outage"] = float(e[w].max())
                rows.append(row)
                np.savez_compressed(out / "logs" / f"{name}_{case}_{f}.npz",
                                    **{k: v for k, v in log.items() if k != "filter"})  # fmt: skip
                print(name, case, f, f"rmse {row['rmse_p']:.3f} m")
            save_curves(
                out / "curves" / f"{name}_{case}.npz",
                t_log,
                curves,
                nav_conditions(rec, t_log),
                platform="euroc",
                set_name=case,
            )
    keys = sorted(
        {k for r in rows for k in r},
        key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99,
    )
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out / 'runs.csv'}")


if __name__ == "__main__":
    main()
