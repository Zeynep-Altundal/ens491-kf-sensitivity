"""IESKF, UKF, InEKF and the heading / standstill (ZUPT + ZARU) updates."""

import numpy as np
import pytest

from experiments.nav_compare.compare.runner import run_filter, summarize
from experiments.nav_compare.data.disturbances import Event, apply_events
from experiments.nav_compare.data.magnetometer import simulate_heading
from experiments.nav_compare.data.odometry import simulate_odometry
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid

NEW = ["IESKF", "UKF", "InEKF"]


@pytest.fixture(scope="module")
def ugv_rec():
    sc = scenario_grid("ugv")["baseline"].replace(outage_s=(), duration_s=40.0)
    seq = generate(sc, seed=5)
    return to_recording(seq, odometry=simulate_odometry(seq, seed=5),
                        magnetometer=simulate_heading(seq, seed=5))  # fmt: skip


@pytest.mark.parametrize("name", NEW)
def test_new_filters_track_with_gps(ugv_rec, name):
    log = run_filter(ugv_rec, name, log_every=10)
    s = summarize(ugv_rec, log)
    assert np.all(np.isfinite(log["p_hat"]))
    assert s["rmse_p"] < 1.0  # GPS std 0.5 m


@pytest.mark.parametrize("name", NEW)
def test_new_filters_close_to_eskf_in_clean_case(ugv_rec, name):
    ref = summarize(ugv_rec, run_filter(ugv_rec, "ESKF"))["rmse_p"]
    got = summarize(ugv_rec, run_filter(ugv_rec, name))["rmse_p"]
    assert got < 1.5 * ref + 0.05


def test_magnetometer_heading_noise_free_matches_truth():
    seq = generate(scenario_grid("uav")["baseline"].replace(duration_s=10.0), seed=1)
    mag = simulate_heading(seq, std_rad=1e-12)
    q = seq.reference.q_wb
    w, x, y, z = q.T
    yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    ref = np.interp(mag["mag_t"], seq.reference.t_s, np.unwrap(yaw))
    d = (mag["mag_heading"] - ref + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(d)) < 1e-3


@pytest.fixture(scope="module")
def static_rec():
    sc = scenario_grid("ugv")["baseline"].replace(
        outage_s=(), duration_s=90.0, stops=((0.0, 30.0), (70.0, 78.0)))  # fmt: skip
    seq = apply_events(generate(sc, seed=2), [Event("outage", "pos", 10.0, 91.0)])
    return to_recording(seq, odometry=simulate_odometry(seq, seed=2),
                        magnetometer=simulate_heading(seq, seed=2))  # fmt: skip


def test_static_calibration_learns_gyro_bias(static_rec):
    log = run_filter(static_rec, "ESKF", nhc_std=1.0, odometry="speed", static=True)
    i = np.searchsorted(log["t"], 29.0)
    bg_hat = log["bg_hat"][i]
    bg_true = np.array([np.interp(29.0, static_rec["truth_t"], static_rec["truth_bg"][:, k])
                        for k in range(3)])  # fmt: skip
    assert np.linalg.norm(bg_hat - bg_true) < 0.2 * np.linalg.norm(bg_true) + 2e-5


def test_static_and_heading_reduce_denied_drift(static_rec):
    def fin(**kw):
        return summarize(static_rec, run_filter(static_rec, "ESKF", nhc_std=1.0, **kw))[
            "final_p"
        ]

    base = fin(odometry="speed")
    assert fin(odometry="speed", static=True) < base
    assert fin(odometry="speed", static=True, magnetometer=True) < base


def test_kf_imu_ignores_heading(ugv_rec):
    a = run_filter(ugv_rec, "KF-IMU", log_every=50)
    b = run_filter(ugv_rec, "KF-IMU", log_every=50, magnetometer=True)
    assert np.array_equal(a["p_hat"], b["p_hat"])
