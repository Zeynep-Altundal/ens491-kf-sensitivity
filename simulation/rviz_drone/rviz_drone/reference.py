"""Reference trajectories with analytic position, velocity and acceleration, and a yaw reference.

Two trajectories, selected by name (``TRAJECTORIES``; ``DEFAULT_TRAJECTORY`` is used when no name is
given):

``mission`` (default, 83 s, area about 42 m x 28 m, cruise height 3 m)
    ground, take-off, hover, straight leg east, coordinated 90 deg left turn, hover with a turn to
    the next heading, one figure-eight, aggressive slalom west, hover with a turn towards home,
    return, hover, landing, ground. The heading follows the direction of travel (in the slalom the
    westward speed used for the heading is at least 2 m/s, so that it is defined at the slow ends).

``figure_eights`` (50 s, the trajectory used until 2026-11-27)
    ground 0-1, take-off 1-5 to 2 m, hover 5-7, figure-eight 7-27, aggressive figure-eight 27-39,
    hover 39-44, landing 44-49, ground 49-50; yaw fixed at 0.

Every segment of the mission starts and ends at rest (minimum-jerk timing along its path), so that
the segments join with continuous position, velocity and acceleration; the heading is the direction
of the path tangent, and changes of heading between segments are made while hovering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

HOVER_Z = 2.0  # [m], figure_eights
MISSION_Z = 3.0  # [m], mission cruise height


@dataclass(frozen=True)
class Segment:
    name: str
    start: float
    end: float


# ---- Building blocks -------------------------------------------------------------------------

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


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _timed_path(g: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]], tau: float,
                length: float, u_end: float):
    """Path g(u), u in [0, u_end], traversed with minimum-jerk timing u(tau) (rest to rest).

    ``g`` returns position, dg/du and d2g/du2. Returns p, v, a and the heading of the tangent dg/du
    (defined also at rest).
    """
    s, ds, dds = min_jerk(tau / length)
    u, du, ddu = u_end * s, u_end * ds / length, u_end * dds / length**2
    p, d1, d2 = g(u)
    return p, d1 * du, d2 * du**2 + d1 * ddu, float(np.arctan2(d1[1], d1[0]))


def _line(p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)

    def g(u):
        return p0 + (p1 - p0) * u, p1 - p0, np.zeros(3)
    return g


def _arc(centre, radius, angle0, z):
    """Counter-clockwise circle (left turn) at height z: angle0 + u is the polar angle."""
    c = np.asarray(centre, float)

    def g(u):
        a = angle0 + u
        p = np.array([c[0] + radius * np.cos(a), c[1] + radius * np.sin(a), z])
        d1 = np.array([-radius * np.sin(a), radius * np.cos(a), 0.0])
        d2 = np.array([-radius * np.cos(a), -radius * np.sin(a), 0.0])
        return p, d1, d2
    return g


def _figure_eight_ns(south_tip, half_length, half_width, z):
    """Figure-eight with its long axis along y, starting and ending at its south tip heading west:
    x = x0 - half_width sin(2u), y = y0 + half_length (1 - cos u)."""
    x0, y0 = south_tip

    def g(u):
        p = np.array([x0 - half_width * np.sin(2 * u), y0 + half_length * (1 - np.cos(u)), z])
        d1 = np.array([-2 * half_width * np.cos(2 * u), half_length * np.sin(u), 0.0])
        d2 = np.array([4 * half_width * np.sin(2 * u), half_length * np.cos(u), 0.0])
        return p, d1, d2
    return g


# ---- Trajectories ----------------------------------------------------------------------------

class Trajectory:
    name: str
    segments: tuple[Segment, ...]

    @property
    def duration(self) -> float:
        return self.segments[-1].end

    def segment_at(self, t: float) -> Segment:
        for s in self.segments:
            if s.start <= t < s.end:
                return s
        return self.segments[-1]

    def reference(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def yaw(self, t: float) -> tuple[float, float]:
        """Yaw reference [rad] and its rate [rad/s]."""
        return 0.0, 0.0


class FigureEights(Trajectory):
    """The trajectory used until 2026-11-27 (unchanged; yaw fixed at 0)."""

    name = "figure_eights"
    segments = (
        Segment("ground", 0.0, 1.0),
        Segment("takeoff", 1.0, 5.0),
        Segment("hover", 5.0, 7.0),
        Segment("figure_eight", 7.0, 27.0),
        Segment("aggressive", 27.0, 39.0),
        Segment("hover", 39.0, 44.0),
        Segment("landing", 44.0, 49.0),
        Segment("ground", 49.0, 50.0),
    )

    @staticmethod
    def _figure_eight(tau, length, period, ax, ay, az, ramp):
        env = envelope(tau, length, ramp)
        w = 2 * np.pi / period
        x = _modulated_sine(ax, w, tau, env)
        y = _modulated_sine(ay, 2 * w, tau, env)
        z = _modulated_sine(az, w, tau, env)
        return (np.array([x[0], y[0], HOVER_Z + z[0]]), np.array([x[1], y[1], z[1]]),
                np.array([x[2], y[2], z[2]]))

    def reference(self, t):
        seg = self.segment_at(t)
        tau, length, zero = t - seg.start, seg.end - seg.start, np.zeros(3)
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
            return self._figure_eight(tau, length, period=10.0, ax=3.0, ay=1.5, az=0.0, ramp=5.0)
        if seg.name == "aggressive":
            return self._figure_eight(tau, length, period=4.0, ax=2.0, ay=0.6, az=0.5, ramp=4.0)
        raise ValueError(seg.name)


class Mission(Trajectory):
    """Readable mission on a larger area; heading along the direction of travel."""

    name = "mission"
    Z = MISSION_Z
    HOME = np.array([0.0, 0.0])
    LEG_END = np.array([30.0, 0.0])  # straight leg east
    TURN_CENTRE, TURN_RADIUS = np.array([30.0, 8.0]), 8.0  # 90 deg left turn to (38, 8), heading north
    EIGHT_TIP, EIGHT_HALF_LENGTH, EIGHT_HALF_WIDTH = np.array([38.0, 8.0]), 10.0, 3.5
    SLALOM_LENGTH, SLALOM_AMPLITUDE, SLALOM_PERIOD, SLALOM_Z_AMPLITUDE = 20.0, 1.5, 2.8, 0.5
    SLALOM_MIN_ALONG = 2.0  # [m/s]
    segments = (
        Segment("ground", 0.0, 1.0),
        Segment("takeoff", 1.0, 5.0),
        Segment("hover", 5.0, 7.0),
        Segment("straight", 7.0, 16.0),
        Segment("turn", 16.0, 22.0),
        Segment("hover_turn", 22.0, 25.0),
        Segment("figure_eight", 25.0, 49.0),
        Segment("aggressive", 49.0, 63.0),
        Segment("hover_turn", 63.0, 66.0),
        Segment("return", 66.0, 74.0),
        Segment("hover", 74.0, 77.0),
        Segment("landing", 77.0, 82.0),
        Segment("ground", 82.0, 83.0),
    )

    def __init__(self):
        z = self.Z
        self._straight = _line([*self.HOME, z], [*self.LEG_END, z])
        self._turn = _arc(self.TURN_CENTRE, self.TURN_RADIUS, -np.pi / 2, z)
        self._eight = _figure_eight_ns(self.EIGHT_TIP, self.EIGHT_HALF_LENGTH, self.EIGHT_HALF_WIDTH, z)
        self.slalom_end = self.EIGHT_TIP + np.array([-self.SLALOM_LENGTH, 0.0])
        self._return = _line([*self.slalom_end, z], [*self.HOME, z])
        self.return_heading = float(np.arctan2(*(self.HOME - self.slalom_end)[::-1]))
        # Heading at the start and end of each moving segment (for the hover turns).
        self.headings = {"straight": 0.0, "turn_end": np.pi / 2, "eight": np.pi, "slalom": np.pi,
                         "return": self.return_heading}

    def _slalom(self, tau: float, length: float):
        s, ds, dds = min_jerk(tau / length)
        env = envelope(tau, length, 2.0)
        w = 2 * np.pi / self.SLALOM_PERIOD
        y = _modulated_sine(self.SLALOM_AMPLITUDE, w, tau, env)
        z = _modulated_sine(self.SLALOM_Z_AMPLITUDE, w * self.SLALOM_PERIOD / 3.5, tau, env)
        x0, y0 = self.EIGHT_TIP
        p = np.array([x0 - self.SLALOM_LENGTH * s, y0 + y[0], self.Z + z[0]])
        v = np.array([-self.SLALOM_LENGTH * ds / length, y[1], z[1]])
        a = np.array([-self.SLALOM_LENGTH * dds / length**2, y[2], z[2]])
        return p, v, a

    def _state(self, t: float):
        """p, v, a and heading [rad] at time t."""
        seg = self.segment_at(t)
        tau, length, zero = t - seg.start, seg.end - seg.start, np.zeros(3)
        z = self.Z
        if seg.name == "ground":
            heading = 0.0 if seg.start == 0.0 else self.return_heading
            return np.array([*self.HOME, 0.0]), zero.copy(), zero.copy(), heading
        if seg.name in ("takeoff", "landing"):
            s, ds, dds = min_jerk(tau / length)
            if seg.name == "landing":
                s, ds, dds = 1.0 - s, -ds, -dds
            heading = 0.0 if seg.name == "takeoff" else self.return_heading
            return (np.array([*self.HOME, z * s]), np.array([0.0, 0.0, z * ds / length]),
                    np.array([0.0, 0.0, z * dds / length**2]), heading)
        if seg.name == "hover":
            heading = 0.0 if seg.start < 10.0 else self.return_heading
            return np.array([*self.HOME, z]), zero.copy(), zero.copy(), heading
        if seg.name == "straight":
            return _timed_path(self._straight, tau, length, 1.0)
        if seg.name == "turn":
            return _timed_path(self._turn, tau, length, np.pi / 2)
        if seg.name == "figure_eight":
            return _timed_path(self._eight, tau, length, 2 * np.pi)
        if seg.name == "aggressive":
            p, v, a = self._slalom(tau, length)
            # Direction of the velocity, with the along-track (westward) speed taken as at least
            # SLALOM_MIN_ALONG m/s, so that the heading stays defined and continuous at the slow ends.
            heading = float(np.arctan2(v[1], min(v[0], -self.SLALOM_MIN_ALONG)))
            return p, v, a, heading
        if seg.name == "return":
            p, v, a, _ = _timed_path(self._return, tau, length, 1.0)
            return p, v, a, self.return_heading
        if seg.name == "hover_turn":
            if seg.start < 30.0:
                pos, h0, h1 = self.EIGHT_TIP, np.pi / 2, np.pi
            else:
                pos, h0, h1 = self.slalom_end, np.pi, self.return_heading
            s, _, _ = min_jerk(tau / length)
            return np.array([*pos, z]), zero.copy(), zero.copy(), h0 + wrap(h1 - h0) * s
        raise ValueError(seg.name)

    def reference(self, t):
        p, v, a, _ = self._state(t)
        return p, v, a

    def yaw(self, t):
        h = 1e-4
        y0 = self._state(t)[3]
        rate = wrap(self._state(t + h)[3] - self._state(t - h)[3]) / (2 * h) if t > h else 0.0
        return y0, float(rate)


TRAJECTORIES: dict[str, Trajectory] = {"mission": Mission(), "figure_eights": FigureEights()}
DEFAULT_TRAJECTORY = "mission"


def get(name: str | None = None) -> Trajectory:
    name = name or DEFAULT_TRAJECTORY
    if name not in TRAJECTORIES:
        raise ValueError(f"unknown trajectory {name!r}; choose from {sorted(TRAJECTORIES)}")
    return TRAJECTORIES[name]


# Module-level functions of the default trajectory (used by the ROS node and the tests).
DURATION = get().duration
SEGMENTS = get().segments


def segment_at(t: float, name: str | None = None) -> Segment:
    return get(name).segment_at(t)


def reference(t: float, name: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference position, velocity and acceleration at time ``t`` (world frame, ENU)."""
    return get(name).reference(t)


def yaw_reference(t: float, name: str | None = None) -> tuple[float, float]:
    return get(name).yaw(t)
