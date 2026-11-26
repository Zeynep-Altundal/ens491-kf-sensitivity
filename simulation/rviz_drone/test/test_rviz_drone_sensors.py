"""Sensor streams generated with synthetic/: hover specific force and reproducibility."""

import numpy as np
import pytest

from rviz_drone import flight
from rviz_drone import quadrotor as qd
from rviz_drone.reference import segment_at
from synthetic.imu_model import ideal_imu
from synthetic.trajectories import TrajectoryState

DURATION = 8.0  # ground, take-off and hover only; keeps the tests fast


def test_ideal_imu_at_hover_state_reads_plus_g_on_body_z():
    params = qd.QuadParams()
    x = qd.hover_state(2.0)
    acc = qd.derivative(x, qd.hover_wrench(params), params)[3:6]
    state = TrajectoryState(t=np.zeros(1), p=x[None, 0:3], v=x[None, 3:6], a=acc[None],
                            q=x[None, 6:10], w_b=x[None, 10:13])
    gyro, accel = ideal_imu(state)
    np.testing.assert_allclose(accel[0], [0.0, 0.0, 9.81], atol=1e-12)
    np.testing.assert_allclose(gyro[0], 0.0, atol=1e-12)


@pytest.fixture(scope="module")
def run_seed0():
    return flight.run(seed=0, duration=DURATION)


def test_imu_in_simulated_hover_reads_plus_g_on_body_z(run_seed0):
    t = run_seed0["t"]
    hover = np.array([segment_at(x).name == "hover" for x in t]) & (t >= 6.0)  # settled
    np.testing.assert_allclose(run_seed0["accel_clean"][hover], np.tile([0.0, 0.0, 9.81], (hover.sum(), 1)),
                               atol=2e-3)
    # The corrupted signal is centred on the clean one plus the true bias.
    resid = run_seed0["accel_meas"][hover] - run_seed0["accel_clean"][hover] - run_seed0["accel_bias"][hover]
    assert np.abs(resid.mean(axis=0)).max() < 0.01


def test_on_the_ground_the_imu_reads_plus_g(run_seed0):
    ground = run_seed0["t"] < 1.0
    np.testing.assert_allclose(run_seed0["accel_clean"][ground], np.tile([0.0, 0.0, 9.81], (ground.sum(), 1)),
                               atol=1e-12)


def test_rates_and_shapes(run_seed0):
    assert run_seed0["t"][1] - run_seed0["t"][0] == pytest.approx(1 / 200)
    assert run_seed0["t_meas"][1] - run_seed0["t_meas"][0] == pytest.approx(1 / 10)
    assert run_seed0["accel_meas"].shape == (int(DURATION * 200) + 1, 3)
    assert run_seed0["z_meas"].shape == (int(DURATION * 10) + 1, 3)


def test_reproducible_from_a_fixed_seed(run_seed0):
    again = flight.run(seed=0, duration=DURATION)
    for key in ("p", "q", "accel_meas", "gyro_meas", "z_meas"):
        np.testing.assert_array_equal(again[key], run_seed0[key])
    other = flight.run(seed=1, duration=DURATION)
    np.testing.assert_array_equal(other["p"], run_seed0["p"])  # the truth does not depend on the seed
    assert not np.array_equal(other["accel_meas"], run_seed0["accel_meas"])
    assert not np.array_equal(other["z_meas"], run_seed0["z_meas"])
