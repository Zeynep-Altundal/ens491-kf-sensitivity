"""Unit tests of the estimators (interface, EKF Jacobian, KF against the linear reference)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.estimators import filters as fl
from experiments.estimators.run import position_epochs, run
from experiments.linear_kf.kf import kalman_filter
from synthetic.rotation import quat_exp, quat_multiply, quat_normalize

TUNING = fl.Tuning(2e-4, 2e-3, 2e-5, 3e-3, 0.5, 0.1, np.radians(1.0), 2e-3, 0.05, 2.0)


def _tangent_basis(q):
    return 0.5 * fl.xi(q)  # columns: dq for unit attitude errors (orthogonal to q)


def test_quaternion_maps_are_consistent():
    q = quat_normalize(np.array([0.9, 0.1, -0.3, 0.2]))
    a = np.array([0.3, -0.2, 0.5])
    np.testing.assert_allclose(quat_multiply(q, np.r_[0.0, a]), fl.xi(q) @ a, atol=1e-15)
    e = quat_exp(np.array([0.01, 0.02, -0.03]))
    np.testing.assert_allclose(quat_multiply(q, e), fl.right_matrix(e) @ q, atol=1e-15)
    np.testing.assert_allclose(quat_multiply(q, e), fl.left_matrix(q) @ e, atol=1e-15)
    A, T = fl.quat_to_att_map(q), fl.att_to_quat_map(q)
    np.testing.assert_allclose(A @ T, np.eye(15), atol=1e-14)


def test_ekf_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = quat_normalize(rng.normal(size=4))
        x = np.r_[rng.normal(size=3), rng.normal(size=3), q, 0.01 * rng.normal(size=3), 0.1 * rng.normal(size=3)]
        g0, g1 = rng.normal(size=3), rng.normal(size=3)
        a0, a1 = rng.normal(size=3) + [0, 0, 9.81], rng.normal(size=3) + [0, 0, 9.81]
        dt = 0.005
        F = fl.ekf_jacobian(x, g0, a0, g1, a1, dt)
        # Directions: every non-quaternion state, and the three tangent directions of q.
        dirs = [np.eye(16)[i] for i in list(range(6)) + list(range(10, 16))]
        for c in _tangent_basis(q).T:
            d = np.zeros(16)
            d[6:10] = c
            dirs.append(d)
        h = 1e-6
        for d in dirs:
            fd = (fl.ekf_step(x + h * d, g0, a0, g1, a1, dt) - fl.ekf_step(x - h * d, g0, a0, g1, a1, dt)) / (2 * h)
            np.testing.assert_allclose(F @ d, fd, atol=2e-7 * max(1.0, np.abs(fd).max()))


def _straight_recording(n=2001, rate=200.0, invalid=()):
    """Level flight at constant velocity, exact IMU, positions every 20 samples."""
    t = np.arange(n) / rate
    v = np.array([1.0, 0.5, 0.0])
    p = t[:, None] * v
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    pos_idx = np.arange(0, n, 20)
    rng = np.random.default_rng(0)
    valid = np.ones(len(pos_idx), bool)
    valid[list(invalid)] = False
    z = p[pos_idx] + 0.5 * rng.normal(size=(len(pos_idx), 3))
    z[~valid] = np.nan
    return {"imu_t": t, "imu_gyro": np.zeros((n, 3)), "imu_accel": np.tile([0, 0, 9.81], (n, 1)),
            "truth_p": p, "truth_v": np.tile(v, (n, 1)), "truth_q": q, "truth_bg": np.zeros((n, 3)),
            "truth_ba": np.zeros((n, 3)), "pos_t": t[pos_idx], "pos_z": z,
            "pos_R": np.tile(0.25 * np.eye(3), (len(pos_idx), 1, 1)), "pos_valid": valid}


@pytest.mark.parametrize("name", ["own_KF", "own_EKF", "own_ESKF"])
def test_interface_and_missing_measurements(name):
    rec = _straight_recording(invalid=range(30, 60))
    out = run(name, rec, TUNING)
    assert out["p_hat"].shape == (2001, 3) and out["P_diag"].shape == (2001, 15)
    assert len(out["nis"]) == int(rec["pos_valid"].sum())
    assert np.all(np.isfinite(out["p_hat"])) and np.all(np.isfinite(out["nees"]))
    # Position variance grows during the outage (no update) and drops after it.
    k = position_epochs(rec["imu_t"], rec["pos_t"])
    var = out["P_diag"][:, 0]
    assert var[k[59] + 19] > var[k[30] - 1] and var[k[61] + 1] < var[k[59] + 19]


def test_kf_matches_the_linear_reference():
    rec = _straight_recording()
    out = run("own_KF", rec, TUNING)
    F, Q = fl.KF.transition(0.1, TUNING.q_cv)
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    x0 = np.r_[rec["truth_p"][0], rec["truth_v"][0]]
    P0 = np.diag(np.r_[np.full(3, 0.25), np.full(3, 0.01)])
    # The reference predicts before each update; the first measurement (t = 0) is applied by the
    # KF without prediction, so the reference starts from the KF posterior at t = 0.
    k = position_epochs(rec["imu_t"], rec["pos_t"])
    f0 = fl.KF(fl.Initial(x0[:3], x0[3:], np.r_[1.0, 0, 0, 0], np.zeros(3), np.zeros(3)), TUNING)
    f0.update_position(rec["pos_z"][0], rec["pos_R"][0])
    ref = kalman_filter(rec["pos_z"][1:], F, H, Q, rec["pos_R"][0], f0.x, f0.P)
    np.testing.assert_allclose(out["p_hat"][k[1:]], ref.estimates[:, :3], atol=1e-9)
    np.testing.assert_allclose(out["v_hat"][k[1:]], ref.estimates[:, 3:], atol=1e-9)


def test_vendor_adapter_passes_the_interval_mean_of_the_imu_samples():
    from experiments.estimators import vendor

    calls = []

    class Fake:
        def predict(self, gyro, accel, dt):
            calls.append((gyro.copy(), accel.copy(), dt))

    est = object.__new__(vendor.VendorEstimator)
    est.f, est.g, est.a = Fake(), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 9.0])
    est.predict(0.005, np.array([0.0, 0.0, 3.0]), np.array([3.0, 0.0, 11.0]))
    est.predict(0.005, np.array([0.0, 0.0, 5.0]), np.array([5.0, 0.0, 13.0]))
    assert vendor.INPUT == "mean"
    np.testing.assert_allclose(calls[0][0], [0, 0, 2.0])
    np.testing.assert_allclose(calls[0][1], [2.0, 0, 10.0])
    np.testing.assert_allclose(calls[1][0], [0, 0, 4.0])
