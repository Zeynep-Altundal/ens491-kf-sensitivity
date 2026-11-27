import numpy as np, sys
from multiprocessing import Pool
from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.disturbances import Event, apply_events
from experiments.nav_compare.data.magnetometer import simulate_heading
from experiments.nav_compare.data.odometry import simulate_odometry
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid

AID = {
    "nhc": {"nhc_std": 1.0},
    "nhc+speed": {"nhc_std": 1.0, "odometry": "speed"},
    "nhc+mag": {"nhc_std": 1.0, "magnetometer": True},
    "nhc+speed+mag": {"nhc_std": 1.0, "odometry": "speed", "magnetometer": True},
}


def job(a):
    seed, grade = a
    sc = scenario_grid("ugv")["baseline"].replace(
        outage_s=(), duration_s=120.0, imu_grade=grade
    )
    seq = apply_events(
        generate(sc, seed=seed), [Event("outage", "pos", 10.0, 121.0)], seed=seed
    )
    rec = to_recording(
        seq,
        odometry=simulate_odometry(seq, seed=seed),
        magnetometer=simulate_heading(seq, seed=seed),
    )
    out = {}
    for k, v in AID.items():
        E = errors(rec, run_filter(rec, "ESKF", **v))
        out[k] = (
            np.linalg.norm(E["e_p"][-1]),
            abs(E["e_th"][-1][2]) if E["e_th"].ndim == 2 else np.nan,
        )
    return seed, grade, out


if __name__ == "__main__":
    jobs = [(s, g) for g in ("medium", "euroc") for s in range(10)]
    with Pool(8) as p:
        res = p.map(job, jobs)
    import collections

    for g in ("medium", "euroc"):
        print("grade", g)
        for k in AID:
            fe = [r[2][k][0] for r in res if r[1] == g]
            th = [np.rad2deg(r[2][k][1]) for r in res if r[1] == g]
            print(
                "  %-14s median %.2f m  mean %.2f  max %.1f   | |yaw err| median %.2f deg"
                % (k, np.median(fe), np.mean(fe), np.max(fe), np.median(th))
            )
        print(
            "  speed better than nhc in",
            sum(r[2]["nhc+speed"][0] < r[2]["nhc"][0] for r in res if r[1] == g),
            "/10;",
            "speed+mag better than mag in",
            sum(
                r[2]["nhc+speed+mag"][0] < r[2]["nhc+mag"][0] for r in res if r[1] == g
            ),
            "/10",
        )
