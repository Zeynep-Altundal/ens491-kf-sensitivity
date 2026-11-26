"""Consistency checks of the quadrotor rigid-body model and of the closed-loop flight."""

import numpy as np
import pytest

from rviz_drone import quadrotor as qd
from rviz_drone.reference import DURATION, segment_at

PARAMS = qd.QuadParams()
DT = 1e-3


def _integrate(x, wrench, seconds, params=PARAMS):
    for _ in range(int(round(seconds / DT))):
        x = qd.rk4_step(x, wrench, params, DT)
    return x


def test_hover_is_an_equilibrium():
    # Thrust m g and zero torque at rest: the state must not move over 10 s.
    x0 = qd.hover_state(2.0)
    wrench = qd.hover_wrench(PARAMS)
    np.testing.assert_allclose(wrench.thrust, PARAMS.mass * 9.81, rtol=1e-12)
    np.testing.assert_allclose(wrench.rotor_thrusts, PARAMS.mass * 9.81 / 4, rtol=1e-12)
    x = _integrate(x0.copy(), wrench, 10.0)
    np.testing.assert_allclose(x, x0, atol=1e-9)


def test_hover_work_energy_balance():
    # Without drag, the change of mechanical energy equals the work done by the thrust. Start in
    # hover with a small upward velocity: thrust m g exactly balances gravity, so the kinetic energy
    # is constant and the potential energy grows by the work m g dz of the thrust.
    params = qd.QuadParams(drag=0.0)
    x = qd.hover_state(2.0)
    x[5] = 0.3
    wrench = qd.hover_wrench(params)
    e0 = 0.5 * params.mass * x[3:6] @ x[3:6] + params.mass * 9.81 * x[2]
    work = 0.0
    for _ in range(5000):
        v_before = x[3:6].copy()
        x_new = qd.rk4_step(x, wrench, params, DT)
        r = qd.quat_to_rotmat(x[6:10])
        work += wrench.thrust * (r @ qd.E3) @ (0.5 * (v_before + x_new[3:6])) * DT
        x = x_new
    e1 = 0.5 * params.mass * x[3:6] @ x[3:6] + params.mass * 9.81 * x[2]
    np.testing.assert_allclose(x[5], 0.3, atol=1e-12)
    np.testing.assert_allclose(e1 - e0, work, rtol=1e-9)


def test_torque_free_rotation_conserves_energy_and_momentum():
    # Free rigid-body rotation (zero torque): rotational kinetic energy and the world-frame angular
    # momentum are invariants of the Euler equations. Zero thrust and a high start, so that the
    # free fall (about 123 m in 5 s) never reaches the ground contact.
    j = PARAMS.J
    x = qd.hover_state(1000.0)
    x[10:13] = [1.0, -0.5, 2.0]
    wrench = qd.Wrench(thrust=0.0, torque=np.zeros(3))

    def invariants(state):
        w = state[10:13]
        r = qd.quat_to_rotmat(state[6:10])
        return 0.5 * w @ j @ w, r @ (j @ w)

    k0, l0 = invariants(x)
    x = _integrate(x, wrench, 5.0)
    k1, l1 = invariants(x)
    np.testing.assert_allclose(k1, k0, rtol=1e-8)
    np.testing.assert_allclose(l1, l0, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(x[6:10]), 1.0, atol=1e-12)


def test_ground_supports_the_vehicle_below_hover_thrust():
    x0 = np.zeros(13)
    x0[6] = 1.0
    x = _integrate(x0.copy(), qd.saturate(PARAMS, 0.9 * PARAMS.mass * 9.81, np.zeros(3)), 1.0)
    np.testing.assert_array_equal(x, x0)


def test_allocation_round_trip():
    thrust, torque = 15.0, np.array([0.1, -0.2, 0.05])
    w = qd.saturate(PARAMS, thrust, torque)
    np.testing.assert_allclose(w.thrust, thrust, rtol=1e-12)
    np.testing.assert_allclose(w.torque, torque, atol=1e-12)
    assert np.all(w.rotor_thrusts >= 0) and np.all(w.rotor_thrusts <= PARAMS.f_max)


@pytest.fixture(scope="module")
def flight_truth():
    from rviz_drone import flight
    return flight.simulate_truth()


def test_flight_tracks_the_reference(flight_truth):
    t, p, p_ref = flight_truth["t"], flight_truth["p"], flight_truth["p_ref"]
    assert t[-1] == pytest.approx(DURATION)
    err = np.linalg.norm(p - p_ref, axis=1)
    names = np.array([segment_at(x).name for x in t])
    assert err[np.isin(names, ["takeoff", "hover", "landing", "ground"])].max() < 0.05
    assert err[names == "figure_eight"].max() < 0.15
    assert err[names == "aggressive"].max() < 1.0
    # Lands where it took off and ends on the ground.
    np.testing.assert_allclose(p[-1], [0.0, 0.0, 0.0], atol=1e-3)
    assert p[:, 2].min() >= 0.0
