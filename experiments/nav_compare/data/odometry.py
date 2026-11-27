"""Simulated vehicle signals of a ground vehicle (UGV): wheel speed and steering angle.

Truth -> measurement:
  speed     v_fwd = x-component of the body velocity R^T v (the car's forward speed);
            measured as (1 + scale_error) * v_fwd + white noise
  steering  bicycle model: yaw rate = v * tan(delta) / L  ->  delta = atan(L * omega_z / v);
            measured as delta + white noise (zero while the car is stopped)

A filter turns the two signals into a forward-speed measurement and a yaw-rate measurement
(omega_z = v_m * tan(delta_m) / L). Units: m/s, rad, m.
"""

from __future__ import annotations

import numpy as np


def simulate_odometry(seq, rate_hz=50.0, speed_std=0.05, speed_scale_error=0.0,
                      steer_std_rad=0.003, wheelbase_m=2.7, seed=0):  # fmt: skip
    if seq.platform != "ugv":
        raise ValueError("odometry is simulated for the UGV only")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    ref = seq.reference
    t_ref = ref.t_s
    t = np.arange(t_ref[0], t_ref[-1], 1.0 / rate_hz)
    q = ref.q_wb
    w, x, y, z = (np.interp(t, t_ref, q[:, i]) for i in range(4))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    # first column of R(q): body x axis in w
    bx = np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1
    )
    v = np.stack([np.interp(t, t_ref, ref.v_w[:, i]) for i in range(3)], axis=1)
    v_fwd = np.einsum("ni,ni->n", bx, v)
    omega_z = np.interp(t, seq.imu.t_s, seq.truth["gyro_clean"][:, 2])
    moving = v_fwd > 0.5
    delta = np.where(
        moving, np.arctan(wheelbase_m * omega_z / np.maximum(v_fwd, 0.5)), 0.0
    )
    speed_m = (1.0 + speed_scale_error) * v_fwd + rng.normal(0.0, speed_std, len(t))
    steer_m = delta + rng.normal(0.0, steer_std_rad, len(t)) * moving
    return {
        "odom_t": t,
        "odom_speed": speed_m,
        "odom_steer": steer_m,
        "odom_meta": {
            "rate_hz": rate_hz,
            "speed_std": speed_std,
            "speed_scale_error": speed_scale_error,
            "steer_std_rad": steer_std_rad,
            "wheelbase_m": wheelbase_m,
        },  # fmt: skip
    }


def yaw_rate_from_odometry(speed, steer, wheelbase_m):
    return speed * np.tan(steer) / wheelbase_m


def yaw_rate_std(speed, steer, speed_std, steer_std, wheelbase_m):
    """First-order standard deviation of the derived yaw rate."""
    d_speed = np.tan(steer) / wheelbase_m
    d_steer = speed / (wheelbase_m * np.cos(steer) ** 2)
    return np.sqrt((d_speed * speed_std) ** 2 + (d_steer * steer_std) ** 2)
