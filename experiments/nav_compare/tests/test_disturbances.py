import numpy as np
import pytest

from experiments.nav_compare.data.disturbances import (
    Event,
    apply_events,
    standard_event_sets,
)
from experiments.nav_compare.data.recording import load_recording, save_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid


@pytest.fixture(scope="module", params=["ugv", "uav"])
def seq(request):
    return generate(scenario_grid(request.param)["baseline"], seed=1)


def _window(seq, a, b):
    t = seq.imu.t_s
    return (t >= a) & (t < b)


def test_truth_series_present(seq):
    tr = seq.truth
    n = len(seq.imu.t_s)
    for k in ("gyro_bias_series", "accel_bias_series", "gyro_clean", "accel_clean"):
        assert tr[k].shape == (n, 3)
    # measured = clean + bias + white noise (no scale factor, no misalignment in baseline)
    resid = seq.imu.accel_m_s2 - tr["accel_clean"] - tr["accel_bias_series"]
    nd = tr["noise_params"]["accel_density"] * np.sqrt(tr["imu_hz"])
    assert abs(resid.std() / nd - 1.0) < 0.05


@pytest.mark.parametrize("sensor,mag", [("gyro", 0.05), ("accel", 1.0)])
def test_imu_noise_burst_only_inside_window(seq, sensor, mag):
    out = apply_events(seq, [Event("noise_burst", sensor, 20.0, 23.0, mag)], seed=3)
    attr = "gyro_rad_s" if sensor == "gyro" else "accel_m_s2"
    d = getattr(out.imu, attr) - getattr(seq.imu, attr)
    w = _window(seq, 20.0, 23.0)
    assert np.all(d[~w] == 0.0)
    assert abs(d[w].std() / mag - 1.0) < 0.1
    other = "accel_m_s2" if sensor == "gyro" else "gyro_rad_s"
    assert np.array_equal(getattr(out.imu, other), getattr(seq.imu, other))
    assert out.truth["events"][-1]["n_samples"] == int(w.sum())


def test_bias_step_updates_truth(seq):
    ev = Event("bias_step", "accel", 15.0, 30.0, 0.2, axes=(1,))
    out = apply_events(seq, [ev])
    w = _window(seq, 15.0, 30.0)
    d = out.imu.accel_m_s2 - seq.imu.accel_m_s2
    assert (
        np.allclose(d[w, 1], 0.2) and np.all(d[~w] == 0) and np.all(d[:, [0, 2]] == 0)
    )
    db = out.truth["accel_bias_series"] - seq.truth["accel_bias_series"]
    assert np.allclose(db, d)  # the truth follows the injected bias exactly


def test_imu_spikes_count_and_log(seq):
    out = apply_events(seq, [Event("spike", "gyro", 10.0, 40.0, 1.0, count=7)], seed=2)
    d = np.abs(out.imu.gyro_rad_s - seq.imu.gyro_rad_s)
    rows = np.flatnonzero(d.max(axis=1) > 0)
    assert len(rows) == 7 and np.allclose(d[rows], 1.0)
    assert np.allclose(sorted(out.truth["events"][-1]["spike_t"]), seq.imu.t_s[rows])


def test_position_events(seq):
    n0 = len(seq.observations)
    out = apply_events(seq, [Event("outage", "pos", 20.0, 30.0)])
    gone = [o for o in seq.observations if 20.0 <= o.t_s < 30.0]
    assert len(out.observations) == n0 - len(gone)
    assert out.hidden_observations == seq.hidden_observations + len(gone)

    out = apply_events(seq, [Event("bias_step", "pos", 20.0, 30.0, 3.0, axes=(0,))])
    for o0, o1 in zip(seq.observations, out.observations, strict=True):
        dz = np.asarray(o1.z) - np.asarray(o0.z)
        want = np.array([3.0, 0, 0]) if 20.0 <= o0.t_s < 30.0 else np.zeros(3)
        assert np.allclose(dz, want)
        assert np.array_equal(o0.r_meas, o1.r_meas)  # the filter is not told

    out = apply_events(seq, [Event("spike", "pos", 10.0, 40.0, 20.0, count=3)], seed=5)
    sizes = [
        np.linalg.norm(np.asarray(a.z) - np.asarray(b.z))
        for a, b in zip(seq.observations, out.observations, strict=True)
    ]
    assert sum(s > 0 for s in sizes) == 3 and np.allclose(
        [s for s in sizes if s > 0], 20.0
    )


def test_deterministic_and_input_unchanged(seq):
    evs = standard_event_sets(seq.platform, seq.imu.t_s[-1])["imu_and_pos_burst"]
    g0 = seq.imu.gyro_rad_s.copy()
    a = apply_events(seq, evs, seed=4)
    b = apply_events(seq, evs, seed=4)
    assert np.array_equal(a.imu.gyro_rad_s, b.imu.gyro_rad_s)
    assert np.array_equal(seq.imu.gyro_rad_s, g0)


def test_event_validation():
    with pytest.raises(ValueError):
        Event("outage", "gyro", 0, 1)
    with pytest.raises(ValueError):
        Event("noise_burst", "pos", 5, 5, 1.0)
    with pytest.raises(ValueError):
        Event("wobble", "pos", 0, 1)


def test_recording_round_trip(seq, tmp_path):
    evs = standard_event_sets(seq.platform, seq.imu.t_s[-1])["pos_noise_burst"]
    out = apply_events(seq, evs, seed=1)
    p = save_recording(out, tmp_path / "rec.npz")
    r = load_recording(p)
    assert r["meta"]["platform"] == seq.platform
    assert np.array_equal(r["imu_gyro"], out.imu.gyro_rad_s)
    assert r["truth_bg"].shape == r["imu_gyro"].shape
    assert r["pos_z"].shape == (len(out.observations), 3)
    assert r["events"][0]["type"] == "noise_burst"
