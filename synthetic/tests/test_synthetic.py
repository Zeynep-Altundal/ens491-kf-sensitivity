"""Tests for the synthetic scenario generator.

Run from the repository root with ``python -m pytest -v -s synthetic`` to see
the measured values.
"""

import dataclasses

import numpy as np
import pytest

from synthetic.imu_model import GRAVITY_W
from synthetic.rotation import quat_exp, quat_multiply, quat_normalize, quat_to_rotmat
from synthetic.scenario import ScenarioConfig, generate, load_npz

DURATION = 60.0
# Final position error allowed for noise-free strapdown integration over 60 s at
# 200 Hz for the profiles with moderate rotation ("straight", "turning").
STRAPDOWN_TOL_M = 0.01
ABSOLUTE_PROFILES = ("straight", "turning")
# For "aggressive" the second-order integrator below has a discretisation error
# of several decimetres at 200 Hz (see synthetic/README.md, "Known
# discretization error"). Instead of an absolute bound, the error must decrease
# at the integrator's order: halving dt must reduce it by about a factor of 4.
CONVERGENCE_PROFILE = "aggressive"
CONVERGENCE_RATES = (100.0, 200.0, 400.0, 800.0)
EXPECTED_ORDER = 2.0
ORDER_TOL = 0.1
NOISE_REL_TOL = 0.05


def strapdown(t, gyro, accel, p0, v0, q0):
    """Integrate body rates and specific force to position.

    Attitude: rotation vector over [t_k, t_k+1] for a linearly varying rate,
    phi = dt/2 (w_k + w_k+1) + dt^2/12 (w_k x w_k+1).
    Velocity and position: exact for a world-frame acceleration that varies
    linearly over each step.
    """
    n = len(t)
    p, v, q = np.array(p0, float), np.array(v0, float), np.array(q0, float)
    a_prev = quat_to_rotmat(q) @ accel[0] + GRAVITY_W
    positions = np.empty((n, 3))
    positions[0] = p
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        phi = 0.5 * dt * (gyro[k] + gyro[k + 1]) + dt**2 / 12.0 * np.cross(gyro[k], gyro[k + 1])
        q = quat_normalize(quat_multiply(q, quat_exp(phi)))
        a_next = quat_to_rotmat(q) @ accel[k + 1] + GRAVITY_W
        p = p + v * dt + dt**2 * (a_prev / 3.0 + a_next / 6.0)
        v = v + 0.5 * dt * (a_prev + a_next)
        a_prev = a_next
        positions[k + 1] = p
    return positions


def _final_strapdown_error(profile, imu_rate):
    out = generate(ScenarioConfig(profile=profile, duration=DURATION, imu_rate=imu_rate))
    pos = strapdown(out["t"], out["gyro_clean"], out["accel_clean"],
                    out["p_true"][0], out["v_true"][0], out["q_true"][0])
    err = np.linalg.norm(pos - out["p_true"], axis=1)
    path_length = np.sum(np.linalg.norm(np.diff(out["p_true"], axis=0), axis=1))
    return err, path_length


@pytest.mark.parametrize("profile", ABSOLUTE_PROFILES)
def test_strapdown_reproduces_ground_truth(profile):
    err, path_length = _final_strapdown_error(profile, 200.0)
    print(f"\n{profile}: final position error after {DURATION:g} s at 200 Hz = {err[-1]:.3e} m, "
          f"max = {err.max():.3e} m, path length = {path_length:.1f} m")
    assert err[-1] < STRAPDOWN_TOL_M


def test_strapdown_error_converges_at_second_order():
    finals = [_final_strapdown_error(CONVERGENCE_PROFILE, rate)[0][-1] for rate in CONVERGENCE_RATES]
    print(f"\n{CONVERGENCE_PROFILE}: final position error after {DURATION:g} s")
    for rate, e in zip(CONVERGENCE_RATES, finals):
        print(f"  {rate:5.0f} Hz: {e:.4e} m")
    orders = [np.log2(a / b) for a, b in zip(finals[:-1], finals[1:])]
    for (r0, r1), e0, e1, order in zip(zip(CONVERGENCE_RATES[:-1], CONVERGENCE_RATES[1:]),
                                       finals[:-1], finals[1:], orders):
        print(f"  {r0:4.0f} -> {r1:4.0f} Hz: error ratio = {e0 / e1:.4f}, observed order = {order:.4f}")
    assert all(abs(order - EXPECTED_ORDER) <= ORDER_TOL for order in orders)


def test_white_noise_std_matches_density():
    cfg = ScenarioConfig(profile="turning", duration=DURATION, enable_turn_on_bias=False,
                         enable_bias_rw=False, enable_imu_outliers=False)
    out = generate(cfg)
    for sensor, density in (("gyro", cfg.gyro_noise_density), ("accel", cfg.accel_noise_density)):
        expected = density * np.sqrt(cfg.imu_rate)  # density / sqrt(dt)
        measured = np.std(out[f"{sensor}_meas"] - out[f"{sensor}_clean"], axis=0)
        ratio = measured / expected
        print(f"\n{sensor}: expected sigma = {expected:.6e}, measured = {measured}, "
              f"ratio = {ratio}")
        assert np.all(np.abs(ratio - 1.0) <= NOISE_REL_TOL)


def _full_error_config(seed):
    return ScenarioConfig(profile="aggressive", duration=20.0, seed=seed,
                          gyro_outlier_prob=1e-3, accel_outlier_prob=1e-3,
                          meas_outlier_prob=0.05, meas_dropout_windows=((5.0, 3.0),))


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"), key


def test_seed_reproducibility():
    first, second = generate(_full_error_config(7)), generate(_full_error_config(7))
    other = generate(_full_error_config(8))
    _assert_same(first, second)
    random_keys = ("gyro_meas", "accel_meas", "gyro_bias", "accel_bias", "z_meas")
    for key in random_keys:
        assert not np.array_equal(first[key], other[key], equal_nan=True), key
    # Ground truth and clean IMU do not depend on the seed.
    for key in ("p_true", "q_true", "gyro_clean", "accel_clean"):
        assert np.array_equal(first[key], other[key]), key


def test_npz_round_trip(tmp_path):
    cfg = _full_error_config(3)
    path = tmp_path / "scenario.npz"
    out = generate(cfg, path)
    loaded, loaded_cfg = load_npz(path)
    assert loaded_cfg == cfg
    _assert_same(out, loaded)


def test_disabling_one_error_source_keeps_others():
    base = _full_error_config(5)
    with_rw = generate(base)
    without_rw = generate(dataclasses.replace(base, enable_bias_rw=False))
    for key in ("gyro_outlier_mask", "accel_outlier_mask", "z_meas"):
        assert np.array_equal(with_rw[key], without_rw[key], equal_nan=True), key
    assert np.allclose(without_rw["gyro_bias"], np.asarray(base.gyro_turn_on_bias))


def test_misalignment_rotates_clean_signals():
    base = ScenarioConfig(profile="aggressive", duration=5.0, enable_white_noise=False,
                          enable_turn_on_bias=False, enable_bias_rw=False)
    body = generate(base)
    sensor = generate(dataclasses.replace(base, imu_misalignment_x_deg=3.0))
    R = sensor["R_bs"]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-15)
    assert np.allclose(sensor["gyro_clean"], body["gyro_clean"] @ R, atol=1e-15)
    assert np.allclose(sensor["accel_clean"], body["accel_clean"] @ R, atol=1e-13)
    # x axis unchanged, norms preserved.
    assert np.array_equal(sensor["gyro_clean"][:, 0], body["gyro_clean"][:, 0])
    assert np.allclose(np.linalg.norm(sensor["accel_clean"], axis=1),
                       np.linalg.norm(body["accel_clean"], axis=1), atol=1e-12)
    zero = generate(dataclasses.replace(base, imu_misalignment_x_deg=0.0))
    assert np.array_equal(zero["R_bs"], np.eye(3))


def test_gyro_scale_factor_scales_sensed_rate_only():
    """With all errors on, (1 + s) acts on the clean rate only; noise, biases and accel unchanged."""
    base = ScenarioConfig(profile="aggressive", duration=5.0)
    zero = generate(base)
    scaled = generate(dataclasses.replace(base, gyro_scale_factor=0.02))
    assert np.array_equal(scaled["gyro_clean"], zero["gyro_clean"])
    assert np.array_equal(scaled["gyro_bias"], zero["gyro_bias"])
    assert np.array_equal(scaled["accel_meas"], zero["accel_meas"])
    assert np.allclose(scaled["gyro_meas"] - zero["gyro_meas"], 0.02 * zero["gyro_clean"],
                       rtol=0.0, atol=1e-15)
    exact_zero = generate(dataclasses.replace(base, gyro_scale_factor=0.0))
    assert np.array_equal(exact_zero["gyro_meas"], zero["gyro_meas"])
