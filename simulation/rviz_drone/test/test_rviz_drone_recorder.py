"""Tests of the flight recorder (common recording format v1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rviz_drone import recorder
from rviz_drone.flight import default_config, run, sensor_streams
from rviz_drone.reference import get
from synthetic.tests.test_synthetic import strapdown

TRAJ = "mission"
STRAPDOWN_WINDOW_S = 1.0
STRAPDOWN_TOL_M = 0.005  # measured 2.0e-3 m (worst window t = 62-63 s, aggressive segment)


@pytest.fixture(scope="module")
def flight0():
    return run(seed=0, trajectory=TRAJ)


@pytest.fixture(scope="module")
def recorded(flight0, tmp_path_factory):
    arrays = recorder.build_recording(TRAJ, 0, flight=flight0)
    path = recorder.write_npz(tmp_path_factory.mktemp("rec") / "mission_seed0.npz", arrays)
    return arrays, path, recorder.load(path)


def test_round_trip_equals_the_source_arrays(recorded):
    arrays, _, rec = recorded
    assert set(rec) == set(arrays)
    for key, a in arrays.items():
        if key in ("meta", "events"):
            continue
        assert rec[key].dtype == a.dtype and rec[key].shape == a.shape, key
        assert np.array_equal(rec[key], a, equal_nan=a.dtype.kind == "f"), key
    import json
    assert rec["meta"] == json.loads(str(arrays["meta"]))
    assert rec["events"] == []


def test_required_keys_shapes_and_metadata(recorded):
    _, _, rec = recorded
    n, k = len(rec["imu_t"]), len(rec["pos_t"])
    duration = get(TRAJ).duration
    assert n == int(round(200 * duration)) + 1 and k == int(round(10 * duration)) + 1
    for key, shape in {"imu_gyro": (n, 3), "imu_accel": (n, 3), "imu_gyro_clean": (n, 3),
                       "imu_accel_clean": (n, 3), "truth_p": (n, 3), "truth_v": (n, 3), "truth_q": (n, 4),
                       "truth_bg": (n, 3), "truth_ba": (n, 3), "pos_z": (k, 3), "pos_R": (k, 3, 3),
                       "pos_valid": (k,)}.items():
        assert rec[key].shape == shape, key
    assert rec["pos_valid"].dtype == bool
    np.testing.assert_allclose(np.linalg.norm(rec["truth_q"], axis=1), 1.0, atol=1e-9)
    m = rec["meta"]
    for key in ("platform", "source", "imu_rate_hz", "pos_rate_hz", "seed", "true_noise", "version",
                "git_commit", "trajectory", "parameters"):
        assert key in m, key
    assert (m["platform"], m["version"], m["trajectory"], m["seed"]) == ("uav", "v1", TRAJ, 0)


def test_timestamps_are_monotonic_at_the_stated_rates(recorded):
    _, _, rec = recorded
    m = rec["meta"]
    for key, rate in (("imu_t", m["imu_rate_hz"]), ("truth_t", m["imu_rate_hz"]), ("pos_t", m["pos_rate_hz"])):
        dt = np.diff(rec[key])
        assert np.all(dt > 0), key
        np.testing.assert_allclose(dt, 1.0 / rate, atol=1e-9, err_msg=key)
    assert rec["imu_t"][0] == 0.0
    assert np.array_equal(rec["truth_t"], rec["imu_t"])
    assert np.array_equal(rec["pos_t"], rec["imu_t"][::20])


def test_recorded_truth_is_consistent_with_the_recorded_imu(recorded):
    _, _, rec = recorded
    t, n = rec["imu_t"], len(rec["imu_t"])
    w = int(round(STRAPDOWN_WINDOW_S * rec["meta"]["imu_rate_hz"]))
    worst = 0.0
    for s in range(0, n - 1, w):
        e = slice(s, min(s + w + 1, n))
        pos = strapdown(t[e], rec["imu_gyro_clean"][e], rec["imu_accel_clean"][e],
                        rec["truth_p"][s], rec["truth_v"][s], rec["truth_q"][s])
        worst = max(worst, float(np.linalg.norm(pos - rec["truth_p"][e], axis=1).max()))
    assert worst < STRAPDOWN_TOL_M
    # Measured minus clean minus true bias is white noise with the configured density.
    noise = rec["meta"]["true_noise"]
    for sensor, bias in (("gyro", "truth_bg"), ("accel", "truth_ba")):
        white = rec[f"imu_{sensor}"] - rec[f"imu_{sensor}_clean"] - rec[bias]
        expected = noise[f"{sensor}_noise_density"] * np.sqrt(rec["meta"]["imu_rate_hz"])
        np.testing.assert_allclose(white.std(axis=0), expected, rtol=0.05)


def test_position_measurements_and_covariance(recorded):
    _, _, rec = recorded
    sigma = np.asarray(rec["meta"]["true_noise"]["pos_sigma"])
    np.testing.assert_array_equal(rec["pos_R"], np.broadcast_to(np.diag(sigma ** 2), rec["pos_R"].shape))
    err = rec["pos_z"][rec["pos_valid"]] - rec["truth_p"][::20][rec["pos_valid"]]
    np.testing.assert_allclose(err.std(axis=0), sigma, rtol=0.1)


def test_same_seed_gives_a_bit_identical_file(recorded, tmp_path):
    _, path, _ = recorded
    again = recorder.write_npz(tmp_path / "again.npz", recorder.build_recording(TRAJ, 0))
    assert again.read_bytes() == path.read_bytes()


def test_other_seed_changes_the_measurements_not_the_truth(flight0):
    config = dataclasses.replace(default_config(seed=1, duration=get(TRAJ).duration))
    other = dict(flight0)
    other.update(sensor_streams(flight0, config))
    a = recorder.build_recording(TRAJ, 0, flight=flight0)
    b = recorder.build_recording(TRAJ, 1, config=config, flight=other)
    assert np.array_equal(a["truth_p"], b["truth_p"]) and np.array_equal(a["truth_q"], b["truth_q"])
    assert not np.array_equal(a["imu_gyro"], b["imu_gyro"])
    assert not np.array_equal(a["pos_z"], b["pos_z"])
