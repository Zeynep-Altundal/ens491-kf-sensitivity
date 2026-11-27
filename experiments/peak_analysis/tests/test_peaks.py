"""Unit tests of the peak rule and of the outage diagnostic helpers."""

from __future__ import annotations

import numpy as np

from experiments.estimators import filters as fl
from experiments.peak_analysis import outage_diagnostic as od
from experiments.peak_analysis.peaks import detect, onsets, segments, threshold

DT = 0.005


def test_threshold_has_a_floor_and_follows_the_nominal_rms():
    assert threshold(np.full(10, 0.01)) == 0.1
    assert np.isclose(threshold(np.full(10, 0.3)), 0.3)


def test_segments_merge_short_gaps_and_drop_short_runs():
    m = np.zeros(1000, bool)
    m[100:130] = True  # 30 epochs: merged with the next run (gap 39 epochs < 40)
    m[169:200] = True
    m[500:539] = True  # 39 epochs alone: too short
    m[700:740] = True  # 40 epochs: kept
    assert segments(m, DT) == [(100, 199), (700, 739)]


def test_detect_attributes_peaks_to_the_latest_onset_before_their_start():
    t = np.arange(2000) * DT
    eps_n = np.full_like(t, 0.2)
    eps_s = eps_n.copy()
    eps_s[600:700] += np.linspace(0.5, 1.0, 100)  # 3.0-3.5 s
    eps_s[100:200] += 1.0  # 0.5-1.0 s, before every onset
    peaks, theta, _ = detect(t, eps_s, eps_n, [("a", 1.0), ("b", 2.5), ("c", 3.2)])
    assert np.isclose(theta, 0.2)
    assert [p.event for p in peaks] == ["", "b"]
    assert np.isclose(peaks[1].t_peak, t[699]) and np.isclose(peaks[1].delay, t[699] - 2.5)
    assert np.isclose(peaks[1].excess, 1.0) and np.isclose(peaks[1].height, 1.2)


def test_onsets_merge_imu_noise_parts_and_split_spikes():
    ev = [{"type": "noise_scale", "sensor": "gyro", "t_start": 5.5, "t_end": 7.5},
          {"type": "noise_scale", "sensor": "accel", "t_start": 5.5, "t_end": 7.5},
          {"type": "spike", "sensor": "pos", "t_start": 1.0, "t_end": 9.0, "times": [3.0, 1.5]}]
    assert onsets(ev) == [("spike@1.5", 1.5), ("spike@3.0", 3.0), ("noise_scale", 5.5)]


def test_outage_window_and_open_loop_propagation_reproduce_the_ekf():
    n, rate = 1201, 200.0
    t = np.arange(n) / rate
    v = np.array([1.0, 0.5, 0.0])
    pos_idx = np.arange(0, n, 20)
    valid = np.ones(len(pos_idx), bool)
    valid[20:40] = False  # 2.0-3.95 s
    rng = np.random.default_rng(3)
    z = (t[pos_idx, None] * v) + 0.1 * rng.normal(size=(len(pos_idx), 3))
    rec = {"imu_t": t, "imu_gyro": 0.01 * rng.normal(size=(n, 3)),
           "imu_accel": np.tile([0, 0, 9.81], (n, 1)) + 0.05 * rng.normal(size=(n, 3)),
           "pos_t": t[pos_idx], "pos_z": z, "pos_R": np.tile(0.01 * np.eye(3), (len(pos_idx), 1, 1)),
           "pos_valid": valid}
    k0, k1 = od.window(rec)
    assert (k0, k1) == (400, 799)
    tun = fl.Tuning(2e-4, 2e-3, 2e-5, 3e-3, 0.5, 0.1, np.radians(1.0), 2e-3, 0.05, 2.0)
    f = fl.EKF(fl.Initial(np.zeros(3), v, np.r_[1.0, 0, 0, 0], rec["imu_gyro"][0], rec["imu_accel"][0]), tun)
    upd = dict(zip(pos_idx.tolist(), range(len(pos_idx))))
    p_hat, x0 = np.empty((n, 3)), None
    for k in range(n):
        if k:
            f.predict(t[k] - t[k - 1], rec["imu_gyro"][k], rec["imu_accel"][k])
        j = upd.get(k)
        if j is not None and valid[j]:
            f.update_position(z[j], rec["pos_R"][j])
        if k == k0 - 1:
            x0 = f.x.copy()
        p_hat[k] = f.x[0:3]
    np.testing.assert_allclose(od.propagate(x0, rec, k0, k1), p_hat[k0:k1 + 1], atol=1e-12)
