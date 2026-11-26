"""Waypoint reference trajectory with analytic position, velocity and acceleration.

Segments (times in s): ground 0-1, take-off 1-5 to 2 m, hover 5-7, figure-eight 7-27,
aggressive figure-eight 27-39, hover 39-44, landing 44-49, ground 49-50.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOVER_Z = 2.0
DURATION = 50.0


@dataclass(frozen=True)
class Segment:
    name: str
    start: float
    end: float


SEGMENTS = (
    Segment("ground", 0.0, 1.0),
    Segment("takeoff", 1.0, 5.0),
    Segment("hover", 5.0, 7.0),
    Segment("figure_eight", 7.0, 27.0),
    Segment("aggressive", 27.0, 39.0),
    Segment("hover", 39.0, 44.0),
    Segment("landing", 44.0, 49.0),
    Segment("ground", 49.0, 50.0),
)


def segment_at(t: float) -> Segment:
    for s in SEGMENTS:
        if s.start <= t < s.end:
            return s
    return SEGMENTS[-1]


def min_jerk(u: float) -> tuple[float, float, float]:
    """s(u) = 10u^3 - 15u^4 + 6u^5 on [0, 1] and its first two derivatives with respect to u."""
    u = min(max(u, 0.0), 1.0)
    return (10 * u**3 - 15 * u**4 + 6 * u**5,
            30 * u**2 - 60 * u**3 + 30 * u**4,
            60 * u - 180 * u**2 + 120 * u**3)


def envelope(tau: float, length: float, ramp: float) -> tuple[float, float, float]:
    """Smooth 0 -> 1 -> 0 envelope over [0, length] with ramps of ``ramp`` seconds."""
    if tau < ramp:
        s, ds, dds = min_jerk(tau / ramp)
        return s, ds / ramp, dds / ramp**2
    if tau > length - ramp:
        s, ds, dds = min_jerk((length - tau) / ramp)
        return s, -ds / ramp, dds / ramp**2
    return 1.0, 0.0, 0.0


def _modulated_sine(amp: float, omega: float, tau: float, env) -> tuple[float, float, float]:
    """amp * e(tau) * sin(omega tau) and its first two time derivatives."""
    e, de, dde = env
    s, c = np.sin(omega * tau), np.cos(omega * tau)
    return (amp * e * s,
            amp * (de * s + e * omega * c),
            amp * (dde * s + 2 * de * omega * c - e * omega**2 * s))


def _figure_eight(tau: float, length: float, period: float, ax: float, ay: float, az: float,
                  ramp: float):
    env = envelope(tau, length, ramp)
    w = 2 * np.pi / period
    x = _modulated_sine(ax, w, tau, env)
    y = _modulated_sine(ay, 2 * w, tau, env)
    z = _modulated_sine(az, w, tau, env)
    p = np.array([x[0], y[0], HOVER_Z + z[0]])
    v = np.array([x[1], y[1], z[1]])
    a = np.array([x[2], y[2], z[2]])
    return p, v, a


def reference(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference position, velocity and acceleration at time ``t`` (world frame, ENU)."""
    seg = segment_at(t)
    tau = t - seg.start
    length = seg.end - seg.start
    zero = np.zeros(3)
    if seg.name == "ground":
        return zero.copy(), zero.copy(), zero.copy()
    if seg.name == "hover":
        return np.array([0.0, 0.0, HOVER_Z]), zero.copy(), zero.copy()
    if seg.name in ("takeoff", "landing"):
        s, ds, dds = min_jerk(tau / length)
        if seg.name == "landing":
            s, ds, dds = 1.0 - s, -ds, -dds
        return (np.array([0.0, 0.0, HOVER_Z * s]), np.array([0.0, 0.0, HOVER_Z * ds / length]),
                np.array([0.0, 0.0, HOVER_Z * dds / length**2]))
    if seg.name == "figure_eight":
        return _figure_eight(tau, length, period=10.0, ax=3.0, ay=1.5, az=0.0, ramp=5.0)
    if seg.name == "aggressive":
        return _figure_eight(tau, length, period=4.0, ax=2.0, ay=0.6, az=0.5, ramp=4.0)
    raise ValueError(seg.name)
