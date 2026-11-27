import numpy as np
import pytest

from experiments.nav_compare.compare.runner import run_filter, summarize
from experiments.nav_compare.data.disturbances import Event, apply_events
from experiments.nav_compare.data.odometry import (
    simulate_odometry,
    yaw_rate_from_odometry,
)
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid


@pytest.fixture(scope="module")
def seq():
    sc = scenario_grid("ugv")["baseline"].replace(outage_s=(), duration_s=60.0)
    return generate(sc, seed=3)


def test_noise_free_odometry_matches_truth(seq):
    od = simulate_odometry(seq, speed_std=1e-12, steer_std_rad=1e-12)
    ref = seq.reference
    sp = np.interp(od["odom_t"], ref.t_s, np.linalg.norm(ref.v_w, axis=1))
    assert (
        np.max(np.abs(od["odom_speed"] - sp)) < 1e-3
    )  # no slip: speed = forward speed
    yr = yaw_rate_from_odometry(od["odom_speed"], od["odom_steer"], 2.7)
    gz = np.interp(od["odom_t"], seq.imu.t_s, seq.truth["gyro_clean"][:, 2])
    moving = od["odom_speed"] > 1.0
    assert np.max(np.abs(yr - gz)[moving]) < 1e-6


def test_odometry_only_for_ugv():
    uav = generate(scenario_grid("uav")["baseline"].replace(duration_s=10.0), seed=0)
    with pytest.raises(ValueError):
        simulate_odometry(uav)


def test_aiding_reduces_gnss_denied_drift(seq):
    denied = apply_events(seq, [Event("outage", "pos", 10.0, 61.0)])
    rec = to_recording(denied, odometry=simulate_odometry(denied, seed=3))
    none = summarize(rec, run_filter(rec, "ESKF"))["final_p"]
    nhc = summarize(rec, run_filter(rec, "ESKF", nhc_std=1.0))["final_p"]
    spd = summarize(rec, run_filter(rec, "ESKF", nhc_std=1.0, odometry="speed"))[
        "final_p"
    ]
    assert nhc < 0.2 * none
    assert (
        spd < 0.2 * none
    )  # which aiding is better is a multi-seed question, not a unit test


def test_kf_cv_ignores_aiding(seq):
    rec = to_recording(seq, odometry=simulate_odometry(seq, seed=3))
    a = run_filter(rec, "KF-CV", log_every=50)
    b = run_filter(rec, "KF-CV", log_every=50, nhc_std=1.0, odometry="both")
    assert np.array_equal(a["p_hat"], b["p_hat"])
