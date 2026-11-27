"""Simulate the full flight, then generate the sensor streams with the models of ``synthetic/``.

The dynamics and the controller run at ``sim_rate`` (1 kHz). Truth and IMU are sampled at the IMU
rate of the ``ScenarioConfig`` (200 Hz), position measurements at its measurement rate (10 Hz).
The random streams are split exactly as in ``synthetic.scenario.generate``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import repo_path  # noqa: F401
from .quadrotor import (ControllerGains, QuadParams, Wrench, control, derivative, rk4_step)
from . import reference as references
from .reference import DURATION
from synthetic.imu_model import corrupt_sensor, ideal_imu
from synthetic.measurements import position_measurements
from synthetic.scenario import RNG_STREAMS, ScenarioConfig, sensor_to_body_rotation
from synthetic.trajectories import TrajectoryState


def default_config(seed: int = 0, duration: float = DURATION) -> ScenarioConfig:
    """Sensor configuration: the defaults of ``ScenarioConfig`` (200 Hz IMU, 10 Hz position)."""
    return ScenarioConfig(duration=duration, seed=seed)


def simulate_truth(duration: float | None = None, sim_rate: float = 1000.0, out_rate: float = 200.0,
                   params: QuadParams = QuadParams(), gains: ControllerGains = ControllerGains(),
                   trajectory: str | None = None) -> dict[str, np.ndarray]:
    """Closed-loop flight of the named trajectory (default: ``reference.DEFAULT_TRAJECTORY``);
    returns truth at ``out_rate`` including the true acceleration and the yaw reference."""
    traj = references.get(trajectory)
    duration = traj.duration if duration is None else duration
    step = int(round(sim_rate / out_rate))
    if abs(sim_rate / out_rate - step) > 1e-9:
        raise ValueError("sim_rate must be an integer multiple of out_rate")
    n_sim = int(round(duration * sim_rate))
    dt = 1.0 / sim_rate
    n_out = n_sim // step + 1

    out = {k: np.zeros((n_out, d)) for k, d in
           (("p", 3), ("v", 3), ("a", 3), ("q", 4), ("w_b", 3), ("p_ref", 3), ("rotor", 4))}
    out["t"] = np.arange(n_out) * step * dt
    out["thrust"] = np.zeros(n_out)
    out["yaw_ref"] = np.zeros(n_out)

    x = np.zeros(13)
    x[6] = 1.0
    wrench = Wrench(thrust=0.0)
    for k in range(n_sim + 1):
        t = k * dt
        p_ref, v_ref, a_ref = traj.reference(t)
        yaw_ref, yaw_rate_ref = traj.yaw(t)
        wrench = control(x, p_ref, v_ref, a_ref, params, gains, yaw_ref, yaw_rate_ref)
        if k % step == 0:
            i = k // step
            xd = derivative(x, wrench, params)
            out["p"][i], out["v"][i], out["q"][i], out["w_b"][i] = x[0:3], x[3:6], x[6:10], x[10:13]
            out["a"][i] = xd[3:6]
            out["p_ref"][i] = p_ref
            out["rotor"][i] = wrench.rotor_thrusts
            out["thrust"][i] = wrench.thrust
            out["yaw_ref"][i] = yaw_ref
        if k < n_sim:
            x = rk4_step(x, wrench, params, dt)
    return out


def sensor_streams(truth: dict[str, np.ndarray], config: ScenarioConfig) -> dict[str, np.ndarray]:
    """IMU and position measurements for a truth sampled at ``config.imu_rate``.

    Same error models, error switches and random-stream split as ``synthetic.scenario.generate``.
    """
    t = truth["t"]
    dt = 1.0 / config.imu_rate
    if abs((t[1] - t[0]) - dt) > 1e-12:
        raise ValueError("truth must be sampled at config.imu_rate")
    state = TrajectoryState(t=t, p=truth["p"], v=truth["v"], a=truth["a"], q=truth["q"],
                            w_b=truth["w_b"])
    gyro_clean, accel_clean = ideal_imu(state)
    r_bs = sensor_to_body_rotation(config.imu_misalignment_x_deg)
    if config.imu_misalignment_x_deg != 0.0:
        gyro_clean = gyro_clean @ r_bs
        accel_clean = accel_clean @ r_bs
    gyro_sensed = gyro_clean
    if config.gyro_scale_factor != 0.0:
        gyro_sensed = (1.0 + config.gyro_scale_factor) * gyro_clean

    children = np.random.SeedSequence(config.seed).spawn(len(RNG_STREAMS))
    rng = {name: np.random.default_rng(ss) for name, ss in zip(RNG_STREAMS, children)}
    gyro = corrupt_sensor(gyro_sensed, dt, config.gyro_model(),
                          rng["gyro_noise"], rng["gyro_bias_rw"], rng["gyro_outliers"])
    accel = corrupt_sensor(accel_clean, dt, config.accel_model(),
                           rng["accel_noise"], rng["accel_bias_rw"], rng["accel_outliers"])

    step = int(round(config.imu_rate / config.meas_rate))
    meas_index = np.arange(0, len(t), step)
    meas = position_measurements(t[meas_index], truth["p"][meas_index], config.measurement_model(),
                                 rng["meas_noise"], rng["meas_outliers"])
    return {
        "gyro_clean": gyro_clean, "accel_clean": accel_clean,
        "gyro_meas": gyro.measured, "accel_meas": accel.measured,
        "gyro_bias": gyro.bias, "accel_bias": accel.bias,
        "t_meas": t[meas_index], "meas_index": meas_index, "z_meas": meas["z"],
        "meas_available": meas["available"], "R_bs": r_bs,
    }


def run(seed: int = 0, duration: float | None = None, config: ScenarioConfig | None = None,
        trajectory: str | None = None) -> dict[str, np.ndarray]:
    """Truth plus sensor streams for one flight; ``config`` overrides the sensor configuration."""
    duration = references.get(trajectory).duration if duration is None else duration
    config = config or default_config(seed=seed, duration=duration)
    if config.duration != duration:
        config = dataclasses.replace(config, duration=duration)
    truth = simulate_truth(duration=duration, out_rate=config.imu_rate, trajectory=trajectory)
    result = dict(truth)
    result.update(sensor_streams(truth, config))
    result["config_json"] = np.array(config.to_json())
    return result


__all__ = ["default_config", "run", "sensor_streams", "simulate_truth"]
