"""Analytic ground-truth trajectories with exact derivatives.

Every profile returns, at arbitrary times ``t``, the position, velocity and
acceleration in the world frame (ENU), the body-to-world attitude quaternion
``q_wb`` (Hamilton, scalar first) and the body angular rate ``w_b``. All
quantities are evaluated in closed form; no numerical differentiation is used.

Profiles
--------
``straight``
    Horizontal straight-line motion along a fixed heading with a repeating
    pattern of constant-acceleration segments (rest, accelerate, cruise,
    decelerate). Consecutive segments are joined by short linear ramps of the
    acceleration so that the acceleration is continuous. The attitude is
    constant (body x axis along the direction of travel).
``turning``
    Planar figure-eight (Gerono-type Lissajous curve) at constant height with
    the yaw aligned with the velocity vector and zero roll and pitch.
``aggressive``
    Three-dimensional Lissajous position with fast, independent sinusoidal
    roll, pitch and yaw motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation import euler_zyx_rates_to_body_rate, euler_zyx_to_quat


@dataclass(frozen=True)
class TrajectoryState:
    """Ground-truth state at the times ``t`` (N,)."""

    t: np.ndarray  # (N,)
    p: np.ndarray  # (N, 3) position in world frame [m]
    v: np.ndarray  # (N, 3) velocity in world frame [m/s]
    a: np.ndarray  # (N, 3) acceleration in world frame [m/s^2]
    q: np.ndarray  # (N, 4) body-to-world quaternion [w, x, y, z]
    w_b: np.ndarray  # (N, 3) body angular rate [rad/s]


def _sinusoid(t: np.ndarray, amp: float, freq: float, phase: float = 0.0):
    """Value and first two derivatives of ``amp * sin(2 pi freq t + phase)``."""
    om = 2.0 * np.pi * freq
    arg = om * t + phase
    return amp * np.sin(arg), amp * om * np.cos(arg), -amp * om**2 * np.sin(arg)


def _state_from_euler(t, p, v, a, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate):
    q = euler_zyx_to_quat(roll, pitch, yaw)
    w_b = euler_zyx_rates_to_body_rate(roll, pitch, roll_rate, pitch_rate, yaw_rate)
    return TrajectoryState(t=t, p=p, v=v, a=a, q=q, w_b=w_b)


@dataclass(frozen=True)
class StraightTrajectory:
    """Straight-line motion with constant-acceleration segments.

    ``segments`` is a repeating pattern of ``(duration [s], acceleration [m/s^2])``.
    The acceleration changes linearly from one segment value to the next over
    ``ramp`` seconds at the start of each segment. When the pattern satisfies
    ``sum(duration * acceleration) == 0`` the speed is periodic.
    """

    segments: tuple[tuple[float, float], ...] = ((7.0, 0.0), (8.0, 0.5), (7.0, 0.0), (8.0, -0.5))
    ramp: float = 1.0
    heading: float = np.deg2rad(30.0)  # yaw of the direction of travel, from east towards north
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "straight"

    def __post_init__(self) -> None:
        if any(d <= self.ramp for d, _ in self.segments):
            raise ValueError("every segment must be longer than the ramp")

    def _knots(self, t_end: float):
        """Knots of the piecewise-linear scalar acceleration with exact v and s at each knot."""
        period = sum(d for d, _ in self.segments)
        n_periods = int(np.floor(t_end / period)) + 1
        times, accels = [0.0], [0.0]  # starts at rest with zero acceleration
        t0, a_prev = 0.0, 0.0
        for _ in range(n_periods):
            for duration, accel in self.segments:
                times += [t0, t0 + self.ramp]
                accels += [a_prev, accel]
                t0 += duration
                a_prev = accel
        times.append(t0)
        accels.append(a_prev)
        times, accels = np.array(times), np.array(accels)
        h = np.diff(times)
        vel = np.zeros_like(times)
        dist = np.zeros_like(times)
        for i, hi in enumerate(h):
            a0, a1 = accels[i], accels[i + 1]
            vel[i + 1] = vel[i] + hi * (a0 + a1) / 2.0
            dist[i + 1] = dist[i] + vel[i] * hi + hi**2 * (a0 / 3.0 + a1 / 6.0)
        return times, accels, vel, dist

    def evaluate(self, t: np.ndarray) -> TrajectoryState:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        times, accels, vel, dist = self._knots(float(np.max(t)))
        # Interval index i such that times[i] <= t < times[i+1]; zero-length intervals are skipped.
        i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
        h = times[i + 1] - times[i]
        s = t - times[i]
        a0 = accels[i]
        jerk = np.where(h > 0, (accels[i + 1] - a0) / np.where(h > 0, h, 1.0), 0.0)
        acc = a0 + jerk * s
        spd = vel[i] + a0 * s + jerk * s**2 / 2.0
        pos = dist[i] + vel[i] * s + a0 * s**2 / 2.0 + jerk * s**3 / 6.0

        direction = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        zeros = np.zeros_like(t)
        return _state_from_euler(
            t,
            np.asarray(self.start) + pos[:, None] * direction,
            spd[:, None] * direction,
            acc[:, None] * direction,
            zeros, zeros, zeros + self.heading,
            zeros, zeros, zeros,
        )


@dataclass(frozen=True)
class TurningTrajectory:
    """Planar figure-eight ``x = A sin(w t)``, ``y = B sin(2 w t)`` with yaw along velocity.

    With these parameters the speed never vanishes, so the yaw is always defined.
    The yaw is unwrapped along the evaluation times to keep the quaternion continuous.
    """

    amp_x: float = 20.0
    amp_y: float = 10.0
    period: float = 30.0
    height: float = 0.0
    name: str = "turning"

    def evaluate(self, t: np.ndarray) -> TrajectoryState:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        f = 1.0 / self.period
        x, vx, ax = _sinusoid(t, self.amp_x, f)
        y, vy, ay = _sinusoid(t, self.amp_y, 2.0 * f)
        zeros = np.zeros_like(t)
        yaw = np.unwrap(np.arctan2(vy, vx))
        yaw_rate = (vx * ay - vy * ax) / (vx**2 + vy**2)
        return _state_from_euler(
            t,
            np.stack([x, y, zeros + self.height], axis=-1),
            np.stack([vx, vy, zeros], axis=-1),
            np.stack([ax, ay, zeros], axis=-1),
            zeros, zeros, yaw,
            zeros, zeros, yaw_rate,
        )


@dataclass(frozen=True)
class AggressiveTrajectory:
    """3D Lissajous position with fast sinusoidal roll, pitch and yaw.

    Each position axis and each Euler angle is ``amp * sin(2 pi freq t + phase)``
    given as ``(amp, freq [Hz], phase [rad])``; the yaw additionally has a
    constant rate ``yaw_drift``. Pitch amplitudes stay well below pi/2.
    """

    pos_x: tuple[float, float, float] = (10.0, 1.0 / 20.0, 0.0)
    pos_y: tuple[float, float, float] = (8.0, 1.0 / 14.0, 0.5)
    pos_z: tuple[float, float, float] = (2.0, 1.0 / 10.0, 0.0)
    roll: tuple[float, float, float] = (0.6, 0.8, 0.0)
    pitch: tuple[float, float, float] = (0.4, 0.6, 0.3)
    yaw: tuple[float, float, float] = (1.0, 0.25, 0.0)
    yaw_drift: float = 0.3
    name: str = "aggressive"

    def evaluate(self, t: np.ndarray) -> TrajectoryState:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        comps = [_sinusoid(t, *c) for c in (self.pos_x, self.pos_y, self.pos_z)]
        p, v, a = (np.stack([c[i] for c in comps], axis=-1) for i in range(3))
        roll, roll_rate, _ = _sinusoid(t, *self.roll)
        pitch, pitch_rate, _ = _sinusoid(t, *self.pitch)
        yaw, yaw_rate, _ = _sinusoid(t, *self.yaw)
        return _state_from_euler(
            t, p, v, a,
            roll, pitch, yaw + self.yaw_drift * t,
            roll_rate, pitch_rate, yaw_rate + self.yaw_drift,
        )


PROFILES = {
    "straight": StraightTrajectory,
    "turning": TurningTrajectory,
    "aggressive": AggressiveTrajectory,
}


def get_trajectory(profile: str):
    """Instantiate a profile with its default parameters."""
    try:
        return PROFILES[profile]()
    except KeyError:
        raise ValueError(f"unknown profile {profile!r}; expected one of {tuple(PROFILES)}") from None
