"""Everything of the replay that does not need Isaac Sim (numpy only, tested in ``tests/``).

Camera paths, propeller angles, the trail behind the drone, the flight phase and the overlay text
are computed here from the simulator truth (``t``, ``p``, ``v``, ``rotor`` at 200 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Prims of the propellers in the ARL-Robot-1 asset, in the rotor order of
# ``QuadParams.rotor_positions``: front-right, rear-left, front-left, rear-right.
ROTOR_PRIMS = ("front_right_prop", "back_left_prop", "front_left_prop", "back_right_prop")

PHASE_LABELS = {"ground": "on ground", "takeoff": "take-off", "hover": "hover",
                "figure_eight": "figure-eight", "aggressive": "aggressive figure-eight",
                "landing": "landing"}

CAMERAS = ("chase", "overview", "topdown")


@dataclass(frozen=True)
class CameraPose:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray


def sample_index(t: np.ndarray, time: float) -> int:
    """Index of the truth sample nearest to ``time`` (uniform sampling)."""
    dt = (t[-1] - t[0]) / (len(t) - 1)
    return int(np.clip(np.round((time - t[0]) / dt), 0, len(t) - 1))


def low_pass(x: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """Causal first-order low-pass filter along axis 0 (exact discretisation), x(0) as initial value."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    alpha = 1.0 - np.exp(-np.diff(t) / tau)
    for k in range(1, len(t)):
        y[k] = y[k - 1] + alpha[k - 1] * (x[k] - y[k - 1])
    return y


def rotor_spin_direction(allocation: np.ndarray) -> np.ndarray:
    """Sign of the rotor angular velocity about body z (+1 counter-clockwise seen from above).

    Row 3 of the allocation matrix is the reaction yaw torque per unit thrust; the rotor spins in
    the opposite direction.
    """
    return -np.sign(allocation[3])


def prop_angles(t: np.ndarray, rotor_thrust: np.ndarray, hover_thrust: float, spin: np.ndarray,
                hover_rev_s: float = 4.0, max_rev_s: float = 7.0) -> np.ndarray:
    """Propeller angles [rad] (n, 4) for display.

    The displayed speed is ``hover_rev_s * sqrt(f / f_hover)`` revolutions per second, limited to
    ``max_rev_s`` so that it stays below half the frame rate at 30 frames/s (no backward-turning
    aliasing). It is a visual cue, not the physical rotor speed (thousands of rpm).
    """
    rate = hover_rev_s * np.sqrt(np.clip(rotor_thrust, 0.0, None) / hover_thrust)
    rate = np.minimum(rate, max_rev_s) * 2.0 * np.pi
    dt = np.diff(t, prepend=t[0])
    return np.cumsum(rate * dt[:, None], axis=0) * spin[None, :]


def chase_path(t: np.ndarray, p: np.ndarray, offset=(-1.7, 1.1, 0.65), tau_eye: float = 0.6,
               tau_target: float = 0.12, min_height: float = 0.25, min_distance: float = 2.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Eye and target of the chase camera: a world-fixed offset, both low-pass filtered.

    The default offset puts the camera on the side of the sun of the sky image (north-west, see
    ``render_flight.py``), so that the drone is lit from the front.
    """
    eye = low_pass(p + np.asarray(offset), t, tau_eye)
    target = low_pass(p + np.array([0.0, 0.0, 0.05]), t, tau_target)
    # When the drone flies towards the lagging camera, push the camera back along the line of sight
    # (the push itself is smoothed with the same time constant).
    los = eye - p
    dist = np.linalg.norm(los, axis=1, keepdims=True)
    push = np.where(dist < min_distance, (min_distance - dist) / np.maximum(dist, 1e-9), 0.0) * los
    eye = eye + low_pass(push, t, tau_eye / 3.0)
    eye[:, 2] = np.maximum(eye[:, 2], min_height)
    return eye, target


# Overview from the north-west (sun side), looking at the centre of the figure-eight.
OVERVIEW = CameraPose(eye=np.array([-5.0, 7.0, 3.6]), target=np.array([0.0, 0.0, 1.2]),
                      up=np.array([0.0, 0.0, 1.0]))
# Top-down: looking along -z with north (+y) up in the image, so east (+x) is to the right.
TOPDOWN = CameraPose(eye=np.array([0.0, 0.0, 11.0]), target=np.array([0.0, 0.0, 0.0]),
                     up=np.array([0.0, 1.0, 0.0]))


def camera_pose(name: str, k: int, chase_eye: np.ndarray, chase_target: np.ndarray) -> CameraPose:
    if name == "chase":
        return CameraPose(eye=chase_eye[k], target=chase_target[k], up=np.array([0.0, 0.0, 1.0]))
    if name == "overview":
        return OVERVIEW
    if name == "topdown":
        return TOPDOWN
    raise ValueError(f"unknown camera {name!r}; choose from {CAMERAS}")


def trail(t: np.ndarray, p: np.ndarray, k: int, length_s: float = 1.5, n: int = 31
          ) -> tuple[np.ndarray, np.ndarray]:
    """Points of the trail behind the drone (oldest first) and their age fraction in [0, 1]."""
    times = t[k] - np.linspace(length_s, 0.0, n)
    times = np.maximum(times, t[0])
    idx = np.array([sample_index(t, x) for x in times])
    age = (t[k] - t[idx]) / length_s
    return p[idx], np.clip(age, 0.0, 1.0)


def overlay_lines(time: float, phase: str, speed: float, camera: str) -> list[str]:
    return [f"t = {time:6.2f} s", f"phase: {PHASE_LABELS.get(phase, phase)}",
            f"speed: {speed:5.2f} m/s", f"camera: {camera}"]
