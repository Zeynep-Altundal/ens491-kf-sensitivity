import numpy as np
from multiprocessing import Pool
from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.disturbances import Event, apply_events
from experiments.nav_compare.data.odometry import simulate_odometry
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid

AID = {"nhc": {"nhc_std": 1.0}, "nhc+speed": {"nhc_std": 1.0, "odometry": "speed"}}
FILTS = ("ESKF", "InEKF")


def job(seed):
    sc = scenario_grid("ugv")["baseline"].replace(outage_s=(), duration_s=120.0)
    seq = apply_events(
        generate(sc, seed=seed), [Event("outage", "pos", 10.0, 121.0)], seed=seed
    )
    rec = to_recording(seq, odometry=simulate_odometry(seq, seed=seed))
    out = {}
    for f in FILTS:
        for k, v in AID.items():
            E = errors(rec, run_filter(rec, f, **v))
            out[(f, k)] = (
                float(np.linalg.norm(E["e_p"][-1])),
                float(np.rad2deg(abs(E["e_th"][-1][2]))),
                float(np.mean(E["nees_att"][-200:])),
            )
    return seed, out


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(job, range(10))
    for f in FILTS:
        for k in AID:
            a = np.array([r[1][(f, k)] for r in res])
            print(
                f"{f:6s} {k:10s} final median {np.median(a[:, 0]):6.2f} m  yaw err median {np.median(a[:, 1]):5.2f} deg  att NEES(end) median {np.median(a[:, 2]):7.1f}"
            )
        print(
            f,
            "speed better than nhc in",
            sum(r[1][(f, "nhc+speed")][0] < r[1][(f, "nhc")][0] for r in res),
            "/10",
        )
