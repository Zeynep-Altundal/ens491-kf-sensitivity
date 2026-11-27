"""Tests of the corruption layer (scenarios added to a recording in the common format)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rviz_drone import corruption, recorder
from rviz_drone.corruption import SCENARIOS, Corruption, Scenario

IMU_KEYS = ("imu_gyro", "imu_accel", "truth_bg", "truth_ba")
POS_KEYS = ("pos_z", "pos_valid")


@pytest.fixture(scope="module")
def nominal(tmp_path_factory):
    path = recorder.record("mission", 0, tmp_path_factory.mktemp("nom") / "mission_seed0.npz")
    return path, recorder.load(path)


@pytest.fixture(scope="module")
def corrupted(nominal):
    path, rec = nominal
    digest = corruption.sha256(path)
    return {name: corruption.apply(rec, sc, digest) for name, sc in SCENARIOS.items()}


def test_no_corruption_reproduces_the_nominal_file_bit_identically(nominal, tmp_path):
    path, rec = nominal
    out = recorder.write_npz(tmp_path / "same.npz", corruption.to_arrays(corruption.apply(rec, Scenario("none", 0))))
    assert out.read_bytes() == path.read_bytes()


def test_scenario_table_has_the_requested_cases():
    assert len(SCENARIOS) == 31
    assert len({s.seed for s in SCENARIOS.values()}) == 31
    outages = sorted(c.t_end - c.t_start for s in SCENARIOS.values() for c in s.corruptions if c.type == "outage")
    assert outages == [5.0, 5.0, 20.0, 20.0, 40.0, 40.0]


def _changed(a, b):
    return ~((a == b) | (np.isnan(a) & np.isnan(b))) if a.dtype.kind == "f" else a != b


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_corruption_equals_its_stored_truth_and_nothing_else_changes(name, nominal, corrupted):
    _, rec = nominal
    c = corrupted[name]
    # Keys of the common format that no corruption touches stay identical.
    for key in ("imu_t", "imu_gyro_clean", "imu_accel_clean", "truth_t", "truth_p", "truth_v", "truth_q",
                "pos_t", "pos_R"):
        assert np.array_equal(c[key], rec[key]), key
    for sensor, bias in (("gyro", "truth_bg"), ("accel", "truth_ba")):
        scale = c[f"corr_{sensor}_noise_scale"]
        white = rec[f"imu_{sensor}"] - rec[f"imu_{sensor}_clean"] - rec[bias]
        expected = rec[f"imu_{sensor}"] + (scale - 1.0)[:, None] * white + c[f"corr_{sensor}_bias"]
        np.testing.assert_allclose(c[f"imu_{sensor}"], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(c[bias], rec[bias] + c[f"corr_{sensor}_bias"], rtol=0, atol=1e-15)
    outage = c["corr_pos_outage"]
    assert np.array_equal(c["pos_valid"], rec["pos_valid"] & ~outage)
    assert np.all(np.isnan(c["pos_z"][outage]))
    keep = ~outage
    np.testing.assert_allclose(c["pos_z"][keep], rec["pos_z"][keep] + c["corr_pos_noise"][keep]
                               + c["corr_pos_spike"][keep], rtol=0, atol=1e-12)
    # Every stored signal is zero (one, false) outside the windows of the scenario.
    t, tp = rec["imu_t"], rec["pos_t"]
    imu_win = np.zeros(len(t), bool)
    pos_win = np.zeros(len(tp), bool)
    for x in SCENARIOS[name].corruptions:
        if x.sensor == "pos":
            pos_win |= (tp >= x.t_start) & (tp < x.t_end)
        else:
            imu_win |= (t >= x.t_start) & ((t < x.t_end) | (x.type == "bias_ramp" and x.hold_until_end))
    for key in ("corr_gyro_bias", "corr_accel_bias"):
        assert not np.any(c[key][~imu_win])
    for key in ("corr_gyro_noise_scale", "corr_accel_noise_scale"):
        assert np.all(c[key][~imu_win] == 1.0)
    for key in ("corr_pos_noise", "corr_pos_spike", "corr_pos_outage", "corr_pos_noise_std"):
        assert not np.any(c[key][~pos_win])
    assert c["meta"]["scenario"]["name"] == name and len(c["events"]) == len(SCENARIOS[name].corruptions)


PHASE_ONSETS = {"hover": (5.0, 7.0), "straight": (7.0, 16.0), "figure_eight": (25.0, 49.0),
                "aggressive": (49.0, 63.0)}


@pytest.mark.parametrize("phase", sorted(PHASE_ONSETS))
def test_onset_variants_start_in_their_phase_with_the_small_magnitudes(phase, corrupted, nominal):
    from rviz_drone.reference import get
    t = nominal[1]["imu_t"]
    lo, hi = PHASE_ONSETS[phase]
    for name, key, value, length in ((f"accel_bias_step_small_onset_{phase}", "corr_accel_bias", 0.2, 10.0),
                                     (f"gyro_bias_step_small_onset_{phase}", "corr_gyro_bias", 0.01, 10.0)):
        c = SCENARIOS[name].corruptions[0]
        assert lo <= c.t_start < hi and c.t_end - c.t_start == length
        assert get("mission").segment_at(c.t_start).name == ("hover" if phase == "hover" else phase)
        b = corrupted[name][key][:, 0]
        win = (t >= c.t_start) & (t < c.t_end)
        assert np.all(b[win] == value) and not np.any(b[~win])
    name = f"imu_noise_burst_x10_onset_{phase}"
    x, rec = corrupted[name], nominal[1]
    c = SCENARIOS[name].corruptions[0]
    assert lo <= c.t_start < hi and c.t_end - c.t_start == 2.0
    win = (t >= c.t_start) & (t < c.t_end)
    for sensor, bias in (("gyro", "truth_bg"), ("accel", "truth_ba")):
        assert np.all(x[f"corr_{sensor}_noise_scale"][win] == 10.0)
        assert np.all(x[f"corr_{sensor}_noise_scale"][~win] == 1.0)
        w0 = rec[f"imu_{sensor}"] - rec[f"imu_{sensor}_clean"] - rec[bias]
        w1 = x[f"imu_{sensor}"] - x[f"imu_{sensor}_clean"] - x[bias]
        np.testing.assert_allclose(w1[win], 10.0 * w0[win], rtol=1e-9, atol=1e-15)
        assert np.array_equal(w1[~win], w0[~win])


def test_bias_steps_and_ramp_have_the_stated_values(corrupted, nominal):
    t = nominal[1]["imu_t"]
    win = (t >= 25.0) & (t < 49.0)
    for name, sensor, axis, value in (("accel_bias_step_small", "accel", 0, 0.2),
                                      ("accel_bias_step_large", "accel", 0, 1.0),
                                      ("gyro_bias_step_small", "gyro", 0, 0.01),
                                      ("gyro_bias_step_large", "gyro", 0, 0.05)):
        b = corrupted[name][f"corr_{sensor}_bias"]
        assert np.all(b[win, axis] == value) and not np.any(b[~win]) and not np.any(b[:, 1:])
    r = corrupted["accel_bias_ramp"]["corr_accel_bias"][:, 0]
    assert np.all(r[t < 7.0] == 0.0) and np.all(r[t >= 74.0] == 0.5)
    mid = (t >= 7.0) & (t < 74.0)
    np.testing.assert_allclose(r[mid], 0.5 * (t[mid] - 7.0) / 67.0, atol=1e-15)


@pytest.mark.parametrize("name, c", [("imu_noise_x3", 3.0), ("imu_noise_x10", 10.0)])
def test_noise_scale_statistics(name, c, corrupted, nominal):
    rec, x = nominal[1], corrupted[name]
    rate = rec["meta"]["imu_rate_hz"]
    for sensor, bias in (("gyro", "truth_bg"), ("accel", "truth_ba")):
        w0 = rec[f"imu_{sensor}"] - rec[f"imu_{sensor}_clean"] - rec[bias]
        w1 = x[f"imu_{sensor}"] - x[f"imu_{sensor}_clean"] - x[bias]
        np.testing.assert_allclose(w1.std(axis=0) / w0.std(axis=0), c, rtol=1e-9)
        density = rec["meta"]["true_noise"][f"{sensor}_noise_density"]
        np.testing.assert_allclose(w1.std(axis=0), c * density * np.sqrt(rate), rtol=0.05)


@pytest.mark.parametrize("name", ["pos_burst_short_calm", "pos_burst_long_calm",
                                  "pos_burst_short_aggressive", "pos_burst_long_aggressive"])
def test_position_burst_statistics(name, corrupted, nominal):
    tp = nominal[1]["pos_t"]
    c = SCENARIOS[name].corruptions[0]
    win = (tp >= c.t_start) & (tp < c.t_end)
    noise = corrupted[name]["corr_pos_noise"][win]
    assert np.all(corrupted[name]["corr_pos_noise_std"][win] == c.magnitude)
    dof = noise.size
    s2 = float(np.sum(noise ** 2))  # zero-mean noise: s2 / sigma^2 ~ chi2(dof)
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], dof)
    assert lo < s2 / c.magnitude ** 2 < hi


def test_spikes_have_the_stated_magnitude_count_spacing_and_window(corrupted, nominal):
    tp = nominal[1]["pos_t"]
    one = corrupted["pos_spike_single"]["corr_pos_spike"]
    idx = np.flatnonzero(np.linalg.norm(one, axis=1))
    assert len(idx) == 1 and tp[idx[0]] == tp[tp >= 12.0][0]
    np.testing.assert_allclose(np.linalg.norm(one[idx], axis=1), 10.0)
    rep = corrupted["pos_spikes_repeated"]
    idx = np.flatnonzero(np.linalg.norm(rep["corr_pos_spike"], axis=1))
    assert len(idx) == 10
    np.testing.assert_allclose(np.linalg.norm(rep["corr_pos_spike"][idx], axis=1), 10.0)
    assert np.all((tp[idx] >= 25.0) & (tp[idx] < 63.0)) and np.all(np.diff(tp[idx]) >= 1.0 - 1e-9)
    assert rep["events"][0]["times"] == [float(x) for x in tp[idx]]


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n.startswith("outage")])
def test_outage_mask_equals_the_window(name, corrupted, nominal):
    tp = nominal[1]["pos_t"]
    c = SCENARIOS[name].corruptions[0]
    win = (tp >= c.t_start) & (tp < c.t_end)
    assert np.array_equal(corrupted[name]["corr_pos_outage"], win)
    assert win.sum() == int(round(10 * (c.t_end - c.t_start)))


def test_same_seed_gives_identical_files_and_the_nominal_file_is_unchanged(nominal, tmp_path):
    path, _ = nominal
    before = corruption.sha256(path)
    a = corruption.write_all(path, tmp_path / "a", names=["pos_spikes_repeated", "pos_burst_long_calm"])
    b = corruption.write_all(path, tmp_path / "b", names=["pos_spikes_repeated", "pos_burst_long_calm"])
    assert [x.read_bytes() for x in a] == [x.read_bytes() for x in b]
    assert corruption.sha256(path) == before
    loaded = recorder.load(a[0])
    assert loaded["meta"]["scenario"]["nominal_sha256"] == before
    assert loaded["meta"]["scenario"]["seed"] == SCENARIOS["pos_spikes_repeated"].seed


def test_other_seed_changes_random_corruptions(nominal):
    rec = nominal[1]
    base = SCENARIOS["pos_spikes_repeated"]
    a = corruption.apply(rec, base)
    b = corruption.apply(rec, Scenario(base.name, base.seed + 1, base.corruptions))
    assert not np.array_equal(a["corr_pos_spike"], b["corr_pos_spike"])


def test_unknown_corruption_type_is_rejected(nominal):
    with pytest.raises(ValueError):
        corruption.apply(nominal[1], Scenario("bad", 0, (Corruption("drift", "pos"),)))
