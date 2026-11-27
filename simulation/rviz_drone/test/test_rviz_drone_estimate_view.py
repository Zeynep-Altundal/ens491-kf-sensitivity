"""Tests of the visual model of the estimator replay (``rviz_drone/estimate_view.py``)."""

import csv

import numpy as np
import pytest

from rviz_drone import estimate_view as ev
from rviz_drone import topics
from rviz_drone.repo_path import repo_root


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return q if w >= 0 else -q, r


def test_covariance_unpack_and_ellipsoid_reconstruct_the_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    P = a @ a.T + 0.1 * np.eye(3)
    iu = np.triu_indices(3)
    P2 = ev.unpack_cov(P[iu][None, :])[0]
    np.testing.assert_allclose(P2, P)
    semi, rot = ev.ellipsoid(P)
    assert np.isclose(np.linalg.det(rot), 1.0)
    np.testing.assert_allclose(rot @ np.diag((semi / ev.SIGMA) ** 2) @ rot.T, P, atol=1e-12)


def test_rotation_to_quaternion():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q, r = _random_rotation(rng)
        np.testing.assert_allclose(ev.rot_to_quat(r), q, atol=1e-12)
    np.testing.assert_allclose(ev.rot_to_quat(np.diag([1.0, -1.0, -1.0])), [0, 1, 0, 0], atol=1e-12)


def test_corruption_windows_merge_imu_noise_and_split_spikes():
    events = [{"type": "noise_scale", "sensor": "gyro", "t_start": 5.5, "t_end": 7.5},
              {"type": "noise_scale", "sensor": "accel", "t_start": 5.5, "t_end": 7.5},
              {"type": "spike", "sensor": "pos", "t_start": 1.0, "t_end": 9.0, "times": [3.0]},
              {"type": "outage", "sensor": "pos", "t_start": 8.0, "t_end": 48.0},
              {"type": "bias_ramp", "sensor": "accel", "t_start": 7.0, "t_end": 74.0, "hold_until_end": True},
              {"type": "noise_scale", "sensor": "gyro", "t_start": 0.0, "t_end": None}]
    w = ev.windows_from_events(events, 83.0)
    assert [(x.kind, x.t0, x.t1) for x in w] == [("noise_scale", 0.0, 83.0), ("spike", 3.0, 3.5),
                                                 ("noise_scale", 5.5, 7.5), ("bias_ramp", 7.0, 83.0),
                                                 ("outage", 8.0, 48.0)]
    assert [x.colour for x in w].count("outage") == 1
    assert ev.active(w, 3.2)[1].name == "position spike"
    assert ev.banner(ev.active(w, 10.0)).startswith("CORRUPTION: IMU noise")
    assert ev.banner([]) == ""


def test_camera_rule_switches_with_hysteresis():
    t = np.arange(0, 20, 0.005)
    truth = np.zeros((len(t), 3))
    d = np.where((t >= 5) & (t < 10), 8.0, np.where((t >= 10) & (t < 11), 4.5, 1.0))
    est = {"kf": np.zeros((len(t), 3)), "ekf": np.stack([d, 0 * d, 0 * d], axis=1)}
    over, far, dmax = ev.camera_modes(t, truth, est)
    assert not over[t < 5].any()
    assert over[(t >= 5) & (t < 12.99)].all()  # 4.5 m keeps it; back after 2 s within 4 m
    assert not over[t >= 13.01].any()
    assert far[np.searchsorted(t, 6.0)] == "ekf" and np.isclose(dmax[np.searchsorted(t, 6.0)], 8.0)
    assert "switch at > 5 m" in ev.camera_reason("ekf", 8.0)
    assert "back to chase" in ev.camera_reason("ekf", 1.0)


def test_trail_ends_at_the_current_sample():
    t = np.arange(0, 10, 0.005)
    p = np.stack([t, 0 * t, 0 * t], axis=1)
    pts = ev.trail(t, p, 1000, length_s=5.0, n=11)
    assert np.isclose(pts[-1, 0], t[1000]) and np.isclose(pts[0, 0], 0.0)
    assert ev.trail(t, p, 0)[0, 0] == 0.0


def test_legend_and_colours_come_from_the_topic_table():
    items = dict(ev.legend())
    assert items["truth"] == topics.COLOURS["truth"] and items["EKF"] == topics.COLOURS["ekf"]
    assert "spare" not in [n for n, _ in ev.legend()] and "spare" not in ev.SHOWN


def test_load_a_results_run_when_present():
    repo = repo_root()
    paths = ev.repo_paths(repo, "pos_spike_single", 0)
    if not all(p.is_file() for p in paths.values()):
        pytest.skip("results run files not generated (experiments/estimators sweep)")
    view = ev.load(repo, "pos_spike_single", 0)
    assert view.truth_p.shape == view.est_p["ekf"].shape and view.est_P["kf"].shape == (len(view.t), 3, 3)
    with paths["peaks"].open() as fh:
        n = sum(1 for r in csv.DictReader(fh) if r["scenario"] == "pos_spike_single" and r["seed"] == "0")
    assert sum(len(v) for v in view.peaks.values()) == n
    assert [w.kind for w in view.windows] == ["spike"]
    assert np.all(np.linalg.eigvalsh(view.est_P["eskf"][::500]) > 0)
    view.est_p["kf"][5, 2] = -2.0
    lines = ev.overlay_lines(view, 5, "chase")
    assert "below ground, z = -2.0 m" in lines[2] and "below" not in lines[3]
